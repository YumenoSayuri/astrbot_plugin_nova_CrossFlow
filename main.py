"""
Nova CrossFlow - 跨聊天指挥官
AstrBot 插件，通过 LLM Tool Calling 让 AI 自动跨群发消息、跨用户私聊、群临时会话。

用户只需用自然语言说：
  "你去XX群发一句XXX"
  "给QQ号123456789发个私聊说XXX"
  "帮我私聊一下XXX 跟他说XXX"
AI 会自动调用对应工具完成发送。
"""

import asyncio

from astrbot.api import llm_tool, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star
from astrbot.core.config.astrbot_config import AstrBotConfig
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


class CrossFlowPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config

    def _get_cfg(self, key: str, default=None):
        """安全读取配置项"""
        try:
            return self.config[key]
        except (KeyError, TypeError):
            return default

    def _check_perm(self, event: AstrMessageEvent) -> bool:
        """检查用户权限"""
        sender_id = str(event.get_sender_id())
        is_admin = event.is_admin()
        allowed = self._get_cfg("allowed_sender_ids", [])
        return check_permission(sender_id, is_admin, allowed)

    def _get_bot(self, event: AstrMessageEvent):
        """从事件中获取bot实例，如果不是aiocqhttp平台则返回None"""
        if isinstance(event, AiocqhttpMessageEvent):
            return event.bot
        return None

    # ==========================================
    # LLM Tool: 向指定群发送消息
    # ==========================================
    @llm_tool(name="crossflow_send_group_message")
    async def tool_send_group_msg(
        self,
        event: AstrMessageEvent,
        group_id: str,
        message: str,
    ) -> str:
        """向指定的QQ群发送一条消息。当用户要求你去某个群发消息、通知某个群、在某个群说话时使用此工具。

        Args:
            group_id(string): 目标QQ群的群号，纯数字字符串。
            message(string): 要发送的消息内容。
        """
        bot = self._get_bot(event)
        if not bot:
            return "错误：当前平台不支持跨聊天发送，仅支持QQ（aiocqhttp）平台。"

        if not self._check_perm(event):
            return "错误：你没有权限使用跨聊天发送功能。"

        if not group_id or not group_id.strip().isdigit():
            return "错误：群号必须是纯数字。"

        group_id = group_id.strip()

        # 白名单检查
        allowed_groups = self._get_cfg("allowed_target_group_ids", [])
        if not check_target_allowed(group_id, "group", allowed_groups, []):
            return f"错误：群 {group_id} 不在允许发送的目标群列表中。"

        # 延迟
        delay = float(self._get_cfg("send_delay", 0.5) or 0.5)
        if delay > 0:
            await asyncio.sleep(delay)

        max_len = int(self._get_cfg("max_text_length", 2000) or 2000)
        result = await send_group_message(
            bot, int(group_id), message, max_length=max_len
        )

        if result["ok"]:
            return f"已成功向群 {group_id} 发送消息。"
        else:
            return f"发送失败：{result.get('error', '未知错误')}"

    # ==========================================
    # LLM Tool: 向指定用户发送私聊消息
    # ==========================================
    @llm_tool(name="crossflow_send_private_message")
    async def tool_send_private_msg(
        self,
        event: AstrMessageEvent,
        user_id: str,
        message: str,
    ) -> str:
        """向指定的QQ用户发送一条私聊消息。当用户要求你私聊某人、给某人发消息、联系某人时使用此工具。如果目标用户不是机器人好友，会自动尝试通过共同群的临时会话发送。

        Args:
            user_id(string): 目标用户的QQ号，纯数字字符串。
            message(string): 要发送的消息内容。
        """
        bot = self._get_bot(event)
        if not bot:
            return "错误：当前平台不支持跨聊天发送，仅支持QQ（aiocqhttp）平台。"

        if not self._check_perm(event):
            return "错误：你没有权限使用跨聊天发送功能。"

        if not user_id or not user_id.strip().isdigit():
            return "错误：QQ号必须是纯数字。"

        user_id = user_id.strip()

        # 白名单检查
        allowed_users = self._get_cfg("allowed_target_user_ids", [])
        if not check_target_allowed(user_id, "private", [], allowed_users):
            return f"错误：用户 {user_id} 不在允许发送的目标用户列表中。"

        # 延迟
        delay = float(self._get_cfg("send_delay", 0.5) or 0.5)
        if delay > 0:
            await asyncio.sleep(delay)

        enable_temp = bool(self._get_cfg("enable_temp_session", True))
        max_len = int(self._get_cfg("max_text_length", 2000) or 2000)

        result = await smart_private_send(
            bot,
            int(user_id),
            message,
            enable_temp_session=enable_temp,
            max_length=max_len,
        )

        if result["ok"]:
            channel = result.get("channel", "私聊")
            return f"已成功通过{channel}向用户 {user_id} 发送消息。"
        else:
            hint = result.get("hint", "")
            error = result.get("error", "未知错误")
            msg = f"发送失败：{error}"
            if hint:
                msg += f"（{hint}）"
            return msg

    # ==========================================
    # LLM Tool: 获取机器人的群列表
    # ==========================================
    @llm_tool(name="crossflow_get_group_list")
    async def tool_get_group_list(
        self,
        event: AstrMessageEvent,
    ) -> str:
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
    # LLM Tool: 查找与目标用户的共同群
    # ==========================================
    @llm_tool(name="crossflow_find_common_group")
    async def tool_find_common_group(
        self,
        event: AstrMessageEvent,
        user_id: str,
    ) -> str:
        """查找机器人与指定QQ用户共同所在的群。当需要判断能否给某人发临时会话、查找共同群时使用此工具。

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

        # 先检查是否是好友
        friend = await is_friend(bot, target_uid)
        if friend:
            return f"用户 {user_id} 是机器人的好友，可以直接发送私聊消息，无需通过临时会话。"

        common_gid = await find_common_group(bot, target_uid)
        if common_gid:
            try:
                info = await bot.call_action(
                    "get_group_info", group_id=common_gid
                )
                group_name = info.get("group_name", "未知")
            except Exception:
                group_name = "未知"

            return (
                f"用户 {user_id} 不是机器人好友，但找到共同群：{group_name} (群号: {common_gid})。"
                f"可以通过此群发送临时会话。"
            )
        else:
            return (
                f"用户 {user_id} 不是机器人好友，且未找到任何共同群。"
                f"无法通过临时会话联系该用户。"
            )

    # ==========================================
    # 手动命令：/跨流帮助
    # ==========================================
    @filter.command("跨流帮助")
    async def cmd_help(self, event: AstrMessageEvent):
        """查看 CrossFlow 帮助信息"""
        help_text = (
            "🌐 Nova CrossFlow 跨聊天指挥官\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "🤖 自然语言模式（推荐）：\n"
            '  直接说 "去群123456发一句XXX"\n'
            '  直接说 "给QQ号789发个私聊说XXX"\n'
            '  直接说 "你在哪些群里？"\n\n'
            "⌨️ 手动命令模式：\n"
            "  /跨群 <群号> <消息>\n"
            "    → 向指定群发送消息\n"
            "  /跨私聊 <QQ号> <消息>\n"
            "    → 向指定用户发私聊\n"
            "    → 非好友自动尝试临时会话\n"
            "  /跨流帮助\n"
            "    → 查看本帮助信息\n\n"
            "ℹ️ 说明：\n"
            "  • 仅管理员或白名单用户可用\n"
            "  • 非好友自动检测共同群发临时会话\n"
            "  • 可在任意群或私聊中指挥发送"
        )
        yield event.plain_result(help_text)

    # ==========================================
    # 手动命令：/跨群 <群号> <消息>
    # ==========================================
    @filter.command("跨群")
    async def cmd_send_group(
        self,
        event: AiocqhttpMessageEvent,
        group_id_str: str = "",
        *msg_parts: str,
    ):
        """向指定群发送消息: /跨群 <群号> <消息内容>"""
        if not self._check_perm(event):
            yield event.plain_result("❌ 你没有权限使用跨聊天功能")
            return

        if not group_id_str:
            yield event.plain_result("❌ 用法：/跨群 <群号> <消息内容>")
            return

        if not group_id_str.isdigit():
            yield event.plain_result("❌ 群号必须是纯数字")
            return

        message = " ".join(msg_parts).strip()
        if not message:
            yield event.plain_result("❌ 请提供要发送的消息内容")
            return

        target_gid = int(group_id_str)

        # 检查白名单
        allowed_groups = self._get_cfg("allowed_target_group_ids", [])
        if not check_target_allowed(group_id_str, "group", allowed_groups, []):
            yield event.plain_result(f"❌ 群 {group_id_str} 不在允许的目标群列表中")
            return

        # 延迟
        delay = float(self._get_cfg("send_delay", 0.5) or 0.5)
        if delay > 0:
            await asyncio.sleep(delay)

        max_len = int(self._get_cfg("max_text_length", 2000) or 2000)
        result = await send_group_message(
            event.bot, target_gid, message, max_length=max_len
        )

        if result["ok"]:
            yield event.plain_result(f"✅ 已向群 {group_id_str} 发送消息")
        else:
            yield event.plain_result(
                f"❌ 发送失败: {result.get('error', '未知错误')}"
            )

    # ==========================================
    # 手动命令：/跨私聊 <QQ号> <消息>
    # ==========================================
    @filter.command("跨私聊")
    async def cmd_send_private(
        self,
        event: AiocqhttpMessageEvent,
        user_id_str: str = "",
        *msg_parts: str,
    ):
        """向指定用户发私聊: /跨私聊 <QQ号> <消息内容>"""
        if not self._check_perm(event):
            yield event.plain_result("❌ 你没有权限使用跨聊天功能")
            return

        if not user_id_str:
            yield event.plain_result("❌ 用法：/跨私聊 <QQ号> <消息内容>")
            return

        if not user_id_str.isdigit():
            yield event.plain_result("❌ QQ号必须是纯数字")
            return

        message = " ".join(msg_parts).strip()
        if not message:
            yield event.plain_result("❌ 请提供要发送的消息内容")
            return

        target_uid = int(user_id_str)

        # 检查白名单
        allowed_users = self._get_cfg("allowed_target_user_ids", [])
        if not check_target_allowed(user_id_str, "private", [], allowed_users):
            yield event.plain_result(f"❌ 用户 {user_id_str} 不在允许的目标用户列表中")
            return

        # 延迟
        delay = float(self._get_cfg("send_delay", 0.5) or 0.5)
        if delay > 0:
            await asyncio.sleep(delay)

        enable_temp = bool(self._get_cfg("enable_temp_session", True))
        max_len = int(self._get_cfg("max_text_length", 2000) or 2000)

        result = await smart_private_send(
            event.bot,
            target_uid,
            message,
            enable_temp_session=enable_temp,
            max_length=max_len,
        )

        if result["ok"]:
            channel = result.get("channel", "私聊")
            yield event.plain_result(
                f"✅ 已通过{channel}向 {user_id_str} 发送消息"
            )
        else:
            hint = result.get("hint", "")
            error = result.get("error", "未知错误")
            msg = f"❌ 发送失败: {error}"
            if hint:
                msg += f"\n💡 {hint}"
            yield event.plain_result(msg)