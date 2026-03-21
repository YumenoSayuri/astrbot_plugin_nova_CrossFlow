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

        # 预先查找共同群（用于非好友的临时会话）
        _common_group_id: Optional[int] = None

        async def _find_temp_session_group():
            nonlocal _common_group_id
            if state.target_type == "private" and _common_group_id is None:
                try:
                    target_uid = int(state.target_id)
                    friend_check = await is_friend(bot, target_uid)
                    if not friend_check:
                        gid = await find_common_group(bot, target_uid)
                        if gid:
                            _common_group_id = gid
                            logger.info(f"[CrossFlow] 找到临时会话通道: group_id={gid}")
                except Exception as e:
                    logger.debug(f"[CrossFlow] 查找共同群失败: {e}")

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
                    # 群消息直接发
                    await AiocqhttpMessageEvent.send_message(
                        bot=bot,
                        message_chain=message,
                        is_group=True,
                        session_id=str(target_id_int),
                    )
                else:
                    # 私聊：先尝试查找共同群
                    await _find_temp_session_group()

                    # 转为 OneBot JSON 格式
                    messages = await AiocqhttpMessageEvent._parse_onebot_json(message)
                    if not messages:
                        logger.warning("[CrossFlow] 消息解析为空，跳过")
                        return

                    # 构建发送参数
                    send_params = {
                        "user_id": target_id_int,
                        "message": messages,
                    }
                    # 如果有共同群，加上 group_id 走临时会话
                    if _common_group_id:
                        send_params["group_id"] = _common_group_id

                    await bot.send_private_msg(**send_params)

                state.intercept_count += 1
                channel = f"临时会话(via群{_common_group_id})" if _common_group_id and state.target_type == "private" else state.target_type
                logger.info(
                    f"[CrossFlow] 消息已重定向: "
                    f"target={channel}:{state.target_id}, "
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
        """关闭输出重定向，恢复正常发送模式。在完成跨聊天发送后调用此工具。注意：请在所有其他工具调用完成后再调用此工具，不要和其他工具同时调用。

        """
        event_id = id(event)
        state = _active_redirects.get(event_id)

        if state:
            # 延迟恢复：等待并行的工具调用完成发送
            await asyncio.sleep(3.0)
            _active_redirects.pop(event_id, None)
            event.send = state.original_send
            logger.info(
                f"[CrossFlow] 输出重定向已关闭（延迟3秒恢复）: "
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
        group_id = str(group_id).strip() if group_id else ""
        if not group_id or not group_id.isdigit():
            return "错误：群号必须是纯数字。"
        allowed_groups = self._get_cfg("allowed_target_group_ids", [])
        if not check_target_allowed(group_id, "group", allowed_groups, []):
            return f"错误：群 {group_id} 不在允许发送的目标群列表中。"
        delay = float(self._get_cfg("send_delay", 0.5) or 0.5)
        if delay > 0:
            await asyncio.sleep(delay)
        max_len = int(self._get_cfg("max_text_length", 2000) or 2000)
        split_enabled = bool(self._get_cfg("split_send_enabled", False))
        split_seg_len = int(self._get_cfg("split_segment_length", 500) or 500)
        split_delay_val = float(self._get_cfg("split_delay", 0.5) or 0.5)
        result = await send_group_message(
            bot, int(group_id), message, max_length=max_len,
            split_send=split_enabled, split_segment_length=split_seg_len, split_delay=split_delay_val,
        )
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
        user_id = str(user_id).strip() if user_id else ""
        if not user_id or not user_id.isdigit():
            return "错误：QQ号必须是纯数字。"
        allowed_users = self._get_cfg("allowed_target_user_ids", [])
        if not check_target_allowed(user_id, "private", [], allowed_users):
            return f"错误：用户 {user_id} 不在允许发送的目标用户列表中。"
        delay = float(self._get_cfg("send_delay", 0.5) or 0.5)
        if delay > 0:
            await asyncio.sleep(delay)
        enable_temp = bool(self._get_cfg("enable_temp_session", True))
        max_len = int(self._get_cfg("max_text_length", 2000) or 2000)
        split_enabled = bool(self._get_cfg("split_send_enabled", False))
        split_seg_len = int(self._get_cfg("split_segment_length", 500) or 500)
        split_delay_val = float(self._get_cfg("split_delay", 0.5) or 0.5)
        result = await smart_private_send(
            bot, int(user_id), message,
            enable_temp_session=enable_temp, max_length=max_len,
            split_send=split_enabled, split_segment_length=split_seg_len, split_delay=split_delay_val,
        )
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
        user_id = str(user_id).strip() if user_id else ""
        if not user_id or not user_id.isdigit():
            return "错误：QQ号必须是纯数字。"
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
    # ==========================================
    # LLM Tool: 查询指定群的成员列表
    # ==========================================
    @llm_tool(name="crossflow_query_group_members")
    async def tool_query_group_members(
        self,
        event: AstrMessageEvent,
        group_id: str,
        search_name: str = "",
    ) -> str:
        """查询指定QQ群的成员列表。可以查询任意群（不限当前群），通过群昵称或QQ名搜索特定成员获取其QQ号。当需要知道某个群里有谁、查找某人的QQ号、确认某人是否在某个群里时使用此工具。

        Args:
            group_id(string): 要查询的群号，纯数字字符串。
            search_name(string): 可选，搜索关键词。填入后只返回群昵称或QQ名包含此关键词的成员，用于快速找人。留空返回全部成员摘要。
        """
        bot = self._get_bot(event)
        if not bot:
            return "错误：当前平台不支持此功能，仅支持QQ（aiocqhttp）平台。"

        group_id = str(group_id).strip() if group_id else ""
        if not group_id or not group_id.isdigit():
            return "错误：群号必须是纯数字。"
        gid_int = int(group_id)

        try:
            members = await bot.call_action("get_group_member_list", group_id=gid_int)
            if not members or not isinstance(members, list):
                return f"错误：无法获取群 {group_id} 的成员列表，可能是机器人不在该群或权限不足。"
        except Exception as e:
            return f"错误：获取群 {group_id} 成员列表失败: {e}"

        # 处理成员数据
        processed = []
        for m in members:
            uid = str(m.get("user_id", ""))
            display_name = m.get("card") or m.get("nickname") or f"用户{uid}"
            username = m.get("nickname") or f"用户{uid}"
            role = m.get("role", "member")
            processed.append({
                "user_id": uid,
                "display_name": display_name,
                "username": username,
                "role": role,
            })

        # 如果有搜索关键词，过滤
        if search_name and search_name.strip():
            keyword = search_name.strip().lower()
            filtered = [
                m for m in processed
                if keyword in m["display_name"].lower()
                or keyword in m["username"].lower()
                or keyword in m["user_id"]
            ]
            if not filtered:
                # 日志精简
                logger.info(f"[CrossFlow] 群 {group_id} 中未找到匹配 '{search_name}' 的成员 (共{len(processed)}人)")
                return f"在群 {group_id} (共{len(processed)}人) 中未找到匹配 '{search_name}' 的成员。"

            lines = []
            for m in filtered:
                role_cn = {"owner": "群主", "admin": "管理", "member": "成员"}.get(m["role"], m["role"])
                lines.append(f"- {m['display_name']}({m['username']}) QQ:{m['user_id']} [{role_cn}]")

            logger.info(f"[CrossFlow] 群 {group_id} 搜索 '{search_name}' 找到 {len(filtered)} 人")
            return f"群 {group_id} 中匹配 '{search_name}' 的成员 ({len(filtered)}人):\n" + "\n".join(lines)

        # 无搜索关键词：返回摘要（日志只显示前5人）
        total = len(processed)
        preview = processed[:5]
        preview_lines = []
        for m in preview:
            role_cn = {"owner": "群主", "admin": "管理", "member": "成员"}.get(m["role"], m["role"])
            preview_lines.append(f"  {m['display_name']}({m['username']}) QQ:{m['user_id']} [{role_cn}]")

        logger.info(f"[CrossFlow] 查询群 {group_id} 成员: 共{total}人，前{len(preview)}人:\n" + "\n".join(preview_lines))

        # 返回给 AI 的是完整列表（AI 需要用来找人）
        import json
        result_data = {
            "group_id": group_id,
            "member_count": total,
            "members": processed,
        }
        return json.dumps(result_data, ensure_ascii=False)

    # ==========================================
    # LLM Tool: 查询指定群的基本信息
    # ==========================================
    @llm_tool(name="crossflow_query_group_info")
    async def tool_query_group_info(
        self,
        event: AstrMessageEvent,
        group_id: str,
    ) -> str:
        """查询指定QQ群的基本信息（群名、群主、人数等）。当需要了解某个群的详细信息时使用此工具。

        Args:
            group_id(string): 要查询的群号，纯数字字符串。
        """
        bot = self._get_bot(event)
        if not bot:
            return "错误：当前平台不支持此功能，仅支持QQ（aiocqhttp）平台。"

        group_id = str(group_id).strip() if group_id else ""
        if not group_id or not group_id.isdigit():
            return "错误：群号必须是纯数字。"
        gid_int = int(group_id)

        try:
            info = await bot.call_action("get_group_info", group_id=gid_int)
            if not info:
                return f"错误：无法获取群 {group_id} 的信息。"
        except Exception as e:
            return f"错误：获取群 {group_id} 信息失败: {e}"

        group_name = info.get("group_name", "未知")
        member_count = info.get("member_count", "?")
        max_member_count = info.get("max_member_count", "?")

        result = (
            f"群信息：\n"
            f"- 群号: {group_id}\n"
            f"- 群名: {group_name}\n"
            f"- 当前人数: {member_count}\n"
            f"- 最大人数: {max_member_count}"
        )

        logger.info(f"[CrossFlow] 查询群信息: {group_id} ({group_name}, {member_count}人)")
        return result
    # ==========================================
    # LLM Tool: 获取任意群的聊天记录
    # ==========================================
    def _parse_msg_to_line(self, msg: dict) -> str:
        """将单条 OneBot 消息转为文本行"""
        from datetime import datetime
        sender = msg.get("sender", {})
        nickname = sender.get("card") or sender.get("nickname", "未知")
        uid = str(sender.get("user_id", ""))
        msg_time = datetime.fromtimestamp(msg.get("time", 0)).strftime("%m-%d %H:%M")
        text_parts = []
        for part in msg.get("message", []):
            ptype = part.get("type", "")
            if ptype == "text":
                t = part.get("data", {}).get("text", "").strip()
                if t:
                    text_parts.append(t)
            elif ptype == "image":
                text_parts.append("[图片]")
            elif ptype == "face":
                text_parts.append("[表情]")
            elif ptype == "at":
                qq = part.get("data", {}).get("qq", "")
                text_parts.append(f"[@{qq}]")
            elif ptype == "record":
                text_parts.append("[语音]")
            elif ptype == "video":
                text_parts.append("[视频]")
            elif ptype == "reply":
                text_parts.append("[回复]")
        text = " ".join(text_parts).strip()
        if text:
            return f"[{msg_time}] {nickname}({uid}): {text}"
        return ""

    async def _summarize_messages(self, lines: list[str], group_id: str) -> str:
        """用独立供应商对超出上限的消息进行总结"""
        provider_id = str(self._get_cfg("history_summary_provider_id", "") or "").strip()
        text_block = "\n".join(lines)
        prompt = (
            f"以下是QQ群{group_id}的一段聊天记录（{len(lines)}条消息），"
            f"请用简洁的中文总结这段对话的主要内容、话题和关键信息：\n\n{text_block}"
        )
        try:
            if provider_id:
                provider = self.context.get_provider_by_id(provider_id)
            else:
                provider = self.context.get_using_provider()
            if not provider:
                return f"[更早的{len(lines)}条消息无法总结：未找到可用的LLM供应商]"
            resp = await provider.text_chat(prompt=prompt)
            summary = resp.completion_text.strip()
            logger.info(f"[CrossFlow] 已总结群 {group_id} 超限的 {len(lines)} 条消息")
            return summary
        except Exception as e:
            logger.error(f"[CrossFlow] 总结消息失败: {e}")
            return f"[更早的{len(lines)}条消息总结失败: {e}]"

    @llm_tool(name="crossflow_get_group_history")
    async def tool_get_group_history(
        self,
        event: AstrMessageEvent,
        group_id: str,
        count: int = 0,
    ) -> str:
        """获取指定QQ群的最近聊天记录（带时间和发言人）。当需要查看某个群最近在聊什么、了解群动态、总结群聊内容时使用此工具。

        Args:
            group_id(string): 要查询的群号，纯数字字符串。
            count(number): 获取的消息条数，0表示使用默认值。超过上限时会自动总结较早的消息。
        """
        bot = self._get_bot(event)
        if not bot:
            return "错误：当前平台不支持此功能，仅支持QQ（aiocqhttp）平台。"

        group_id = str(group_id).strip() if group_id else ""
        if not group_id or not group_id.isdigit():
            return "错误：群号必须是纯数字。"

        default_count = int(self._get_cfg("history_default_count", 30) or 30)
        max_count = int(self._get_cfg("history_max_count", 200) or 200)
        actual_count = int(count) if count and int(count) > 0 else default_count
        gid_int = int(group_id)

        try:
            ret = await bot.call_action(
                "get_group_msg_history",
                group_id=gid_int,
                message_seq=0,
                count=actual_count,
                reverseOrder=False,
            )
            messages = ret.get("messages", [])
            if not messages:
                return f"群 {group_id} 没有获取到任何聊天记录。"

            chat_lines = []
            for msg in messages:
                line = self._parse_msg_to_line(msg)
                if line:
                    chat_lines.append(line)

            if not chat_lines:
                return f"群 {group_id} 的最近 {actual_count} 条消息中没有有效文本内容。"

            # 超限处理
            if len(chat_lines) > max_count:
                recent_lines = chat_lines[-max_count:]
                older_lines = chat_lines[:-max_count]
                summary = await self._summarize_messages(older_lines, group_id)
                preview = recent_lines[-3:] if len(recent_lines) > 3 else recent_lines
                logger.info(f"[CrossFlow] 获取群 {group_id} 聊天记录: 共{len(chat_lines)}条, 返回最近{len(recent_lines)}条+总结{len(older_lines)}条")
                return (
                    f"群 {group_id} 共获取 {len(chat_lines)} 条消息。\n\n"
                    f"【更早的 {len(older_lines)} 条消息总结】\n{summary}\n\n"
                    f"【最近 {len(recent_lines)} 条消息原文】\n" + "\n".join(recent_lines)
                )
            else:
                preview = chat_lines[-3:] if len(chat_lines) > 3 else chat_lines
                logger.info(f"[CrossFlow] 获取群 {group_id} 聊天记录: 共{len(chat_lines)}条")
                return f"群 {group_id} 最近 {len(chat_lines)} 条聊天记录：\n" + "\n".join(chat_lines)

        except Exception as e:
            logger.error(f"[CrossFlow] 获取群 {group_id} 聊天记录失败: {e}")
            return f"获取群 {group_id} 聊天记录失败: {e}"

    # ==========================================
    # LLM Tool: 获取任意群中指定用户的消息
    # ==========================================
    @llm_tool(name="crossflow_get_user_messages")
    async def tool_get_user_messages(
        self,
        event: AstrMessageEvent,
        group_id: str,
        user_id: str,
        count: int = 0,
    ) -> str:
        """获取指定QQ群中某个用户的最近聊天记录。当需要查看某人在某个群里说了什么、了解某人的发言内容时使用此工具。会从群聊历史中过滤出该用户的消息。

        Args:
            group_id(string): 要查询的群号，纯数字字符串。
            user_id(string): 要查询的用户QQ号，纯数字字符串。
            count(number): 从群历史中扫描的消息总数（不是用户消息数），0表示使用默认值。扫描越多找到的用户消息越多。
        """
        bot = self._get_bot(event)
        if not bot:
            return "错误：当前平台不支持此功能，仅支持QQ（aiocqhttp）平台。"

        group_id = str(group_id).strip() if group_id else ""
        user_id = str(user_id).strip() if user_id else ""
        if not group_id or not group_id.isdigit():
            return "错误：群号必须是纯数字。"
        if not user_id or not user_id.isdigit():
            return "错误：用户QQ号必须是纯数字。"

        default_count = int(self._get_cfg("history_default_count", 30) or 30)
        max_count = int(self._get_cfg("history_max_count", 200) or 200)
        actual_count = int(count) if count and int(count) > 0 else default_count
        gid_int = int(group_id)
        target_uid = user_id

        try:
            ret = await bot.call_action(
                "get_group_msg_history",
                group_id=gid_int,
                message_seq=0,
                count=actual_count,
                reverseOrder=False,
            )
            messages = ret.get("messages", [])
            if not messages:
                return f"群 {group_id} 没有获取到任何聊天记录。"

            user_lines = []
            for msg in messages:
                sender = msg.get("sender", {})
                msg_uid = str(sender.get("user_id", ""))
                if msg_uid != target_uid:
                    continue
                line = self._parse_msg_to_line(msg)
                if line:
                    user_lines.append(line)

            if not user_lines:
                return f"在群 {group_id} 最近 {actual_count} 条消息中没有找到用户 {user_id} 的发言。"

            # 超限处理
            if len(user_lines) > max_count:
                recent = user_lines[-max_count:]
                older = user_lines[:-max_count]
                summary = await self._summarize_messages(older, group_id)
                logger.info(f"[CrossFlow] 用户 {user_id} 在群 {group_id}: 共{len(user_lines)}条, 返回最近{len(recent)}条+总结{len(older)}条")
                return (
                    f"用户 {user_id} 在群 {group_id} 共提取 {len(user_lines)} 条发言。\n\n"
                    f"【更早的 {len(older)} 条发言总结】\n{summary}\n\n"
                    f"【最近 {len(recent)} 条发言原文】\n" + "\n".join(recent)
                )

            logger.info(f"[CrossFlow] 用户 {user_id} 在群 {group_id}: 扫描{len(messages)}条, 找到{len(user_lines)}条")
            return (
                f"用户 {user_id} 在群 {group_id} 的最近发言（从{len(messages)}条群消息中提取了{len(user_lines)}条）：\n"
                + "\n".join(user_lines)
            )

        except Exception as e:
            logger.error(f"[CrossFlow] 获取用户 {user_id} 在群 {group_id} 的消息失败: {e}")
            return f"获取失败: {e}"


    # ==========================================
    # LLM Tool: 跨群禁言
    # ==========================================
    @llm_tool(name="crossflow_ban_member")
    async def tool_ban_member(
        self,
        event: AstrMessageEvent,
        group_id: str,
        user_id: str,
        duration: int = 600,
    ) -> str:
        """在指定QQ群中禁言某个用户。当用户要求禁言某人、让某人闭嘴时使用此工具。需要机器人在该群有管理员权限。

        Args:
            group_id(string): 群号，纯数字字符串。
            user_id(string): 要禁言的用户QQ号，纯数字字符串。
            duration(number): 禁言时长（秒），范围0-2592000（30天），0表示取消禁言。默认600秒（10分钟）。
        """
        bot = self._get_bot(event)
        if not bot:
            return "错误：当前平台不支持此功能，仅支持QQ（aiocqhttp）平台。"
        has_perm, perm_reason = self._check_perm(event)
        if not has_perm:
            return perm_reason
        group_id = str(group_id).strip() if group_id else ""
        user_id = str(user_id).strip() if user_id else ""
        if not group_id or not group_id.isdigit():
            return "错误：群号必须是纯数字。"
        if not user_id or not user_id.isdigit():
            return "错误：用户QQ号必须是纯数字。"

        gid = int(group_id)
        uid = int(user_id)
        dur = max(0, min(2592000, int(duration)))

        try:
            await bot.set_group_ban(group_id=gid, user_id=uid, duration=dur)
            if dur == 0:
                logger.info(f"[CrossFlow] 已解除禁言: 群{gid} 用户{uid}")
                return f"已解除群 {group_id} 中用户 {user_id} 的禁言。"
            else:
                logger.info(f"[CrossFlow] 已禁言: 群{gid} 用户{uid} {dur}秒")
                return f"已在群 {group_id} 中禁言用户 {user_id}，时长 {dur} 秒。"
        except Exception as e:
            logger.error(f"[CrossFlow] 禁言失败: 群{gid} 用户{uid}, {e}")
            return f"禁言失败：{e}。可能是机器人不是该群管理员，或目标用户权限更高。"

    # ==========================================
    # LLM Tool: 跨群踢人
    # ==========================================
    @llm_tool(name="crossflow_kick_member")
    async def tool_kick_member(
        self,
        event: AstrMessageEvent,
        group_id: str,
        user_id: str,
        reject_add: str = "false",
    ) -> str:
        """将指定用户从指定QQ群中踢出。当用户要求踢人、移除某人时使用此工具。需要机器人在该群有管理员权限。

        Args:
            group_id(string): 群号，纯数字字符串。
            user_id(string): 要踢出的用户QQ号，纯数字字符串。
            reject_add(string): 是否拒绝此人再次加群，"true"或"false"，默认"false"。
        """
        bot = self._get_bot(event)
        if not bot:
            return "错误：当前平台不支持此功能，仅支持QQ（aiocqhttp）平台。"
        has_perm, perm_reason = self._check_perm(event)
        if not has_perm:
            return perm_reason
        group_id = str(group_id).strip() if group_id else ""
        user_id = str(user_id).strip() if user_id else ""
        if not group_id or not group_id.isdigit():
            return "错误：群号必须是纯数字。"
        if not user_id or not user_id.isdigit():
            return "错误：用户QQ号必须是纯数字。"

        gid = int(group_id)
        uid = int(user_id)
        reject = reject_add.strip().lower() == "true"

        try:
            await bot.set_group_kick(group_id=gid, user_id=uid, reject_add_request=reject)
            reject_text = "（已拒绝再次加群）" if reject else ""
            logger.info(f"[CrossFlow] 已踢出: 群{gid} 用户{uid} {reject_text}")
            return f"已将用户 {user_id} 从群 {group_id} 中踢出{reject_text}。"
        except Exception as e:
            logger.error(f"[CrossFlow] 踢人失败: 群{gid} 用户{uid}, {e}")
            return f"踢人失败：{e}。可能是机器人不是该群管理员，或目标用户权限更高。"

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