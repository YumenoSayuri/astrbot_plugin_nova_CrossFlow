"""
Nova CrossFlow v1.1.0 - 跨聊天指挥官
AstrBot 插件，通过 LLM Tool Calling 让 AI 自动跨群发消息、跨用户私聊、群临时会话。
支持输出重定向：AI 可以把任何工具的输出（图片/语音/表情包等）重定向到其他群/私聊。

核心原理：通过猴子补丁 event.send() 方法实现输出重定向，
能拦截所有通过 event.send() 发送的消息（包括直接调用的插件工具）。
"""

import asyncio
import time
import types
from typing import Optional

from astrbot.api import llm_tool, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star
from astrbot.core.config.astrbot_config import AstrBotConfig
from astrbot.core.message.message_event_result import MessageChain
from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import (
    AiocqhttpMessageEvent,
)

from .utils import (
    check_permission,
    check_target_allowed,
    find_common_group,
    get_bot_group_list,
    get_friend_list,
    is_friend,
    send_group_message,
    send_private_message,
    smart_private_send,
)


# ==========================================
# 重定向状态管理
# ==========================================
class RedirectState:
    """存储单个 event 的重定向状态"""
    def __init__(
        self,
        target_type: str,
        target_id: str,
        original_send,
        created_at: float,
    ):
        self.target_type = target_type
        self.target_id = target_id
        self.original_send = original_send
        self.created_at = created_at
        self.intercept_count = 0

    def is_expired(self, timeout_seconds: float = 120.0) -> bool:
        return (time.time() - self.created_at) > timeout_seconds


REDIRECT_TIMEOUT_SECONDS = 120.0

# event id -> RedirectState
_active_redirects: dict[int, RedirectState] = {}


def _get_session_key(event: AstrMessageEvent) -> str:
    gid = event.get_group_id()
    uid = event.get_sender_id()
    if gid:
        return f"group_{gid}_sender_{uid}"
    return f"private_{uid}"


class CrossFlowPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config

    def _get_cfg(self, key: str, default=None):
        try:
            return self.config[key]
        except (KeyError, TypeError):
            return default

    def _check_perm(self, event: AstrMessageEvent) -> tuple[bool, str]:
        sender_id = str(event.get_sender_id())
        is_admin = event.is_admin()
        allowed = self._get_cfg("allowed_sender_ids", [])
        has_perm = check_permission(sender_id, is_admin, allowed)
        if has_perm:
            return True, ""
        reason = (
            f"用户 {sender_id} 不在 CrossFlow 插件的发送者白名单中，"
            f"且不是 AstrBot 管理员。请将该用户的QQ号添加到插件配置的"
            f"'允许使用跨聊天功能的QQ号白名单'中，或将其添加为 AstrBot 管理员。"
        )
        logger.warning(f"[CrossFlow] 权限拒绝: sender_id={sender_id}, is_admin={is_admin}")
        return False, reason

    def _get_bot(self, event: AstrMessageEvent):
        if isinstance(event, AiocqhttpMessageEvent):
            return event.bot
        return None

    # ==========================================
    # 核心：猴子补丁 event.send() 实现重定向
    # ==========================================
    def _patch_event_send(
        self,
        event: AiocqhttpMessageEvent,
        target_type: str,
        target_id: str,
    ) -> RedirectState:
        """替换 event.send 方法，让后续所有发送都重定向到目标"""

        original_send = event.send
        bot = event.bot
        state = RedirectState(
            target_type=target_type,
            target_id=target_id,
            original_send=original_send,
            created_at=time.time(),
        )
        event_id = id(event)

        async def redirected_send(message: MessageChain):
            """被重定向的 send 方法"""
            # 超时检查
            if state.is_expired(REDIRECT_TIMEOUT_SECONDS):
                logger.warning("[CrossFlow] 重定向已超时，恢复正常发送")
                _restore_event_send(event, state)
                return await original_send(message)

            try:
                target_id_int = int(state.target_id)
                if state.target_type == "group":
                    await AiocqhttpMessageEvent.send_message(
                        bot=bot,
                        message_chain=message,
                        is_group=True,
                        session_id=str(target_id_int),
                    )
                else:
                    await AiocqhttpMessageEvent.send_message(
                        bot=bot,
                        message_chain=message,
                        is_group=False,
                        session_id=str(target_id_int),
                    )

                state.intercept_count += 1
                logger.info(
                    f"[CrossFlow] 消息已重定向: "
                    f"target={state.target_type}:{state.target_id}, "
                    f"count={state.intercept_count}"
                )
            except Exception as e:
                logger.error(f"[CrossFlow] 重定向发送失败: {e}，回退到原始发送")
                await original_send(message)

        # 替换 send 方法
        event.send = redirected_send
        _active_redirects[event_id] = state
        return state

    # ==========================================
    # LLM Tool: 开启输出重定向
    # ==========================================
    @llm_tool(name="crossflow_redirect_output")
    async def tool_redirect_output(
        self,
        event: AstrMessageEvent,
        target_type: str,
        target_id: str,
    ) -> str:
        """开启输出重定向。调用此工具后，你后续调用的所有其他工具（画图、发表情包、生成语音等）的输出都会被自动重定向到指定的群或私聊，而不是发到当前会话。当用户要求你"去XX群发个表情包"、"给某人私聊发张图"等需要把工具生成内容发到其他地方时，先调用此工具设置重定向目标，然后再调用对应的工具生成内容。完成后记得调用 crossflow_stop_redirect 取消重定向。

        Args:
            target_type(string): 目标类型，必须是 "group"（群聊）或 "private"（私聊）。
            target_id(string): 目标ID，群号或QQ号，纯数字字符串。
        """
        if not isinstance(event, AiocqhttpMessageEvent):
            return "错误：当前平台不支持重定向，仅支持QQ（aiocqhttp）平台。"

        has_perm, perm_reason = self._check_perm(event)
        if not has_perm:
            return perm_reason

        if target_type not in ("group", "private"):
            return "错误：target_type 必须是 'group' 或 'private'。"

        if not target_id or not target_id.strip().isdigit():
            return "错误：target_id 必须是纯数字。"

        target_id = target_id.strip()

        allowed_groups = self._get_cfg("allowed_target_group_ids", [])
        allowed_users = self._get_cfg("allowed_target_user_ids", [])
        if not check_target_allowed(target_id, target_type, allowed_groups, allowed_users):
            return f"错误：{target_type} {target_id} 不在允许的目标列表中。"

        # 如果已有重定向，先恢复
        event_id = id(event)
        if event_id in _active_redirects:
            old_state = _active_redirects.pop(event_id)
            event.send = old_state.original_send

        # 应用猴子补丁
        state = self._patch_event_send(event, target_type, target_id)

        target_name = "群" if target_type == "group" else "用户"
        logger.info(f"[CrossFlow] 输出重定向已开启: target={target_type}:{target_id}")
        return (
            f"输出重定向已开启。现在你调用任何工具生成的内容（文字/图片/语音/表情包等）"
            f"都会自动发送到{target_name} {target_id}，而不是当前会话。"
            f"完成后请调用 crossflow_stop_redirect 取消重定向。"
            f"重定向将在 {int(REDIRECT_TIMEOUT_SECONDS)} 秒后自动取消。"
        )

    # ==========================================
    # LLM Tool: 关闭输出重定向
    # ==========================================
    @llm_tool(name="crossflow_stop_redirect")
    async def tool_stop_redirect(
        self,
        event: AstrMessageEvent,
    ) -> str:
        """关闭输出重定向，恢复正常发送模式。在完成跨聊天发送后调用此工具。

        """
        event_id = id(event)
        state = _active_redirects.pop(event_id, None)

        if state:
            event.send = state.original_send
            logger.info(
                f"[CrossFlow] 输出重定向已关闭: "
                f"共拦截 {state.intercept_count} 条消息"
            )
            return f"输出重定向已关闭。共有 {state.intercept_count} 条消息被重定向发送。现在恢复正常发送模式。"
        else:
            return "当前没有活跃的输出重定向。"

    # ==========================================
    # on_decorating_result Hook (双保险)
    # ==========================================
    @filter.on_decorating_result(priority=5)
    async def on_decorating_result(self, event: AstrMessageEvent):
        """双保险：对于通过 set_result 管道发送的消息也进行重定向"""
        event_id = id(event)
        state = _active_redirects.get(event_id)

        if not state:
            return
        if state.is_expired(REDIRECT_TIMEOUT_SECONDS):
            _active_redirects.pop(event_id, None)
            if hasattr(state, 'original_send'):
                event.send = state.original_send
            return

        if not isinstance(event, AiocqhttpMessageEvent):
            return

        result = event.get_result()
        if not result or not result.chain:
            return

        bot = event.bot
        target_id_int = int(state.target_id)

        try:
            messages = await AiocqhttpMessageEvent._parse_onebot_json(result)
            if not messages:
                return

            if state.target_type == "group":
                await bot.send_group_msg(group_id=target_id_int, message=messages)
            else:
                await bot.send_private_msg(user_id=target_id_int, message=messages)

            state.intercept_count += 1
            result.chain.clear()
            event.set_result(event.plain_result(""))
            logger.info(
                f"[CrossFlow] on_decorating_result 重定向成功: "
                f"target={state.target_type}:{state.target_id}"
            )
        except Exception as e:
            logger.error(f"[CrossFlow] on_decorating_result 重定向失败: {e}")

    # ==========================================
    # LLM Tool: 向指定群发送文本消息
    # ==========================================
    @llm_tool(name="crossflow_send_group_message")
    async def tool_send_group_msg(self, event: AstrMessageEvent, group_id: str, message: str) -> str:
        """向指定的QQ群发送一条文本消息。当用户要求你去某个群发消息、通知某个群、在某个群说话时使用此工具。

        Args:
            group_id(string): 目标QQ群的群号，纯数字字符串。
            message(string): 要发送的消息内容。
        """
        bot = self._get_bot(event)
        if not bot:
            return "错误：当前平台不支持跨聊天发送，仅支持QQ（aiocqhttp）平台。"
        has_perm, perm_reason = self._check_perm(event)
        if not has_perm:
            return perm_reason
        if not group_id or not group_id.strip().isdigit():
            return "错误：群号必须是纯数字。"
        group_id = group_id.strip()
        allowed_groups = self._get_cfg("allowed_target_group_ids", [])
        if not check_target_allowed(group_id, "group", allowed_groups, []):
            return f"错误：群 {group_id} 不在允许发送的目标群列表中。"
        delay = float(self._get_cfg("send_delay", 0.5) or 0.5)
        if delay > 0:
            await asyncio.sleep(delay)
        max_len = int(self._get_cfg("max_text_length", 2000) or 2000)
        result = await send_group_message(bot, int(group_id), message, max_length=max_len)
        if result["ok"]:
            return f"已成功向群 {group_id} 发送消息。"
        return f"发送失败：{result.get('error', '未知错误')}"

    # ==========================================
    # LLM Tool: 向指定用户发送私聊消息
    # ==========================================
    @llm_tool(name="crossflow_send_private_message")
    async def tool_send_private_msg(self, event: AstrMessageEvent, user_id: str, message: str) -> str:
        """向指定的QQ用户发送一条私聊文本消息。当用户要求你私聊某人、给某人发消息、联系某人时使用此工具。如果目标不是好友会自动尝试临时会话。

        Args:
            user_id(string): 目标用户的QQ号，纯数字字符串。
            message(string): 要发送的消息内容。
        """
        bot = self._get_bot(event)
        if not bot:
            return "错误：当前平台不支持跨聊天发送，仅支持QQ（aiocqhttp）平台。"
        has_perm, perm_reason = self._check_perm(event)
        if not has_perm:
            return perm_reason
        if not user_id or not user_id.strip().isdigit():
            return "错误：QQ号必须是纯数字。"
        user_id = user_id.strip()
        allowed_users = self._get_cfg("allowed_target_user_ids", [])
        if not check_target_allowed(user_id, "private", [], allowed_users):
            return f"错误：用户 {user_id} 不在允许发送的目标用户列表中。"
        delay = float(self._get_cfg("send_delay", 0.5) or 0.5)
        if delay > 0:
            await asyncio.sleep(delay)
        enable_temp = bool(self._get_cfg("enable_temp_session", True))
        max_len = int(self._get_cfg("max_text_length", 2000) or 2000)
        result = await smart_private_send(bot, int(user_id), message, enable_temp_session=enable_temp, max_length=max_len)
        if result["ok"]:
            channel = result.get("channel", "私聊")
            return f"已成功通过{channel}向用户 {user_id} 发送消息。"
        hint = result.get("hint", "")
        error = result.get("error", "未知错误")
        msg = f"发送失败：{error}"
        if hint:
            msg += f"（{hint}）"
        return msg

    # ==========================================
    # LLM Tool: 获取群列表
    # ==========================================
    @llm_tool(name="crossflow_get_group_list")
    async def tool_get_group_list(self, event: AstrMessageEvent) -> str:
        """获取机器人当前加入的所有QQ群列表。当用户问机器人在哪些群、查看群列表、需要知道群号时使用此工具。

        """
        bot = self._get_bot(event)
        if not bot:
            return "错误：当前平台不支持此功能，仅支持QQ（aiocqhttp）平台。"
        groups = await get_bot_group_list(bot)
        if not groups:
            return "机器人当前未加入任何群。"
        groups.sort(key=lambda x: x.get("group_id", 0))
        lines = []
        for g in groups:
            gid = g.get("group_id", "?")
            name = g.get("group_name", "未知")
            count = g.get("member_count", "?")
            lines.append(f"- {name} (群号: {gid}, {count}人)")
        return f"机器人加入了 {len(groups)} 个群：\n" + "\n".join(lines)

    # ==========================================
    # LLM Tool: 查找共同群
    # ==========================================
    @llm_tool(name="crossflow_find_common_group")
    async def tool_find_common_group(self, event: AstrMessageEvent, user_id: str) -> str:
        """查找机器人与指定QQ用户共同所在的群。

        Args:
            user_id(string): 目标用户的QQ号，纯数字字符串。
        """
        bot = self._get_bot(event)
        if not bot:
            return "错误：当前平台不支持此功能，仅支持QQ（aiocqhttp）平台。"
        if not user_id or not user_id.strip().isdigit():
            return "错误：QQ号必须是纯数字。"
        user_id = user_id.strip()
        target_uid = int(user_id)
        friend = await is_friend(bot, target_uid)
        if friend:
            return f"用户 {user_id} 是机器人的好友，可以直接发送私聊消息。"
        common_gid = await find_common_group(bot, target_uid)
        if common_gid:
            try:
                info = await bot.call_action("get_group_info", group_id=common_gid)
                group_name = info.get("group_name", "未知")
            except Exception:
                group_name = "未知"
            return f"找到共同群：{group_name} (群号: {common_gid})。可通过临时会话联系。"
        return f"用户 {user_id} 不是好友，且未找到共同群。无法联系。"

    # ==========================================
    # 手动命令
    # ==========================================
    @filter.command("跨流帮助")
    async def cmd_help(self, event: AstrMessageEvent):
        """查看 CrossFlow 帮助信息"""
        help_text = (
            "🌐 Nova CrossFlow v1.1.0\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "🤖 自然语言（推荐）：\n"
            '  "去群123456发一句XXX"\n'
            '  "去XX群发个表情包"\n'
            '  "给QQ号789私聊发张图"\n\n'
            "⌨️ 手动命令：\n"
            "  /跨群 <群号> <消息>\n"
            "  /跨私聊 <QQ号> <消息>\n"
            "  /跨流帮助"
        )
        yield event.plain_result(help_text)

    @filter.command("跨群")
    async def cmd_send_group(self, event: AiocqhttpMessageEvent, group_id_str: str = "", *msg_parts: str):
        has_perm, perm_reason = self._check_perm(event)
        if not has_perm:
            yield event.plain_result(f"❌ {perm_reason}")
            return
        if not group_id_str or not group_id_str.isdigit():
            yield event.plain_result("❌ 用法：/跨群 <群号> <消息内容>")
            return
        message = " ".join(msg_parts).strip()
        if not message:
            yield event.plain_result("❌ 请提供要发送的消息内容")
            return
        allowed_groups = self._get_cfg("allowed_target_group_ids", [])
        if not check_target_allowed(group_id_str, "group", allowed_groups, []):
            yield event.plain_result(f"❌ 群 {group_id_str} 不在允许的目标群列表中")
            return
        delay = float(self._get_cfg("send_delay", 0.5) or 0.5)
        if delay > 0:
            await asyncio.sleep(delay)
        max_len = int(self._get_cfg("max_text_length", 2000) or 2000)
        result = await send_group_message(event.bot, int(group_id_str), message, max_length=max_len)
        if result["ok"]:
            yield event.plain_result(f"✅ 已向群 {group_id_str} 发送消息")
        else:
            yield event.plain_result(f"❌ 发送失败: {result.get('error', '未知错误')}")

    @filter.command("跨私聊")
    async def cmd_send_private(self, event: AiocqhttpMessageEvent, user_id_str: str = "", *msg_parts: str):
        has_perm, perm_reason = self._check_perm(event)
        if not has_perm:
            yield event.plain_result(f"❌ {perm_reason}")
            return
        if not user_id_str or not user_id_str.isdigit():
            yield event.plain_result("❌ 用法：/跨私聊 <QQ号> <消息内容>")
            return
        message = " ".join(msg_parts).strip()
        if not message:
            yield event.plain_result("❌ 请提供要发送的消息内容")
            return
        allowed_users = self._get_cfg("allowed_target_user_ids", [])
        if not check_target_allowed(user_id_str, "private", [], allowed_users):
            yield event.plain_result(f"❌ 用户 {user_id_str} 不在允许的目标用户列表中")
            return
        delay = float(self._get_cfg("send_delay", 0.5) or 0.5)
        if delay > 0:
            await asyncio.sleep(delay)
        enable_temp = bool(self._get_cfg("enable_temp_session", True))
        max_len = int(self._get_cfg("max_text_length", 2000) or 2000)
        result = await smart_private_send(event.bot, int(user_id_str), message, enable_temp_session=enable_temp, max_length=max_len)
        if result["ok"]:
            channel = result.get("channel", "私聊")
            yield event.plain_result(f"✅ 已通过{channel}向 {user_id_str} 发送消息")
        else:
            hint = result.get("hint", "")
            error = result.get("error", "未知错误")
            msg = f"❌ 发送失败: {error}"
            if hint:
                msg += f"\n💡 {hint}"
            yield event.plain_result(msg)


def _restore_event_send(event: AstrMessageEvent, state: RedirectState):
    """恢复 event 的原始 send 方法"""
    event_id = id(event)
    _active_redirects.pop(event_id, None)
    if state and state.original_send:
        event.send = state.original_send