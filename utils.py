"""
Nova CrossFlow - 跨聊天工具函数
包含：跨群发送、跨用户私聊、临时会话自动检测共同群组
"""

import asyncio
from typing import Optional

from aiocqhttp import CQHttp
from astrbot.api import logger
from astrbot.core.message.components import Plain
from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import (
    AiocqhttpMessageEvent,
)


def _split_text_segments(text: str, max_segment_length: int = 500) -> list[str]:
    """将长文本按段落/句号分割成多段

    Args:
        text: 原始文本
        max_segment_length: 每段最大字符数

    Returns:
        分段后的文本列表
    """
    if len(text) <= max_segment_length:
        return [text]

    segments = []
    # 先按换行符分段
    paragraphs = text.split("\n")
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 1 <= max_segment_length:
            current = current + "\n" + para if current else para
        else:
            if current:
                segments.append(current.strip())
            # 如果单个段落超长，按句号再切
            if len(para) > max_segment_length:
                sentences = para.replace("。", "。\n").replace("！", "！\n").replace("？", "？\n").split("\n")
                sub_current = ""
                for s in sentences:
                    if len(sub_current) + len(s) <= max_segment_length:
                        sub_current += s
                    else:
                        if sub_current:
                            segments.append(sub_current.strip())
                        sub_current = s
                current = sub_current
            else:
                current = para

    if current and current.strip():
        segments.append(current.strip())

    return segments if segments else [text]


async def send_group_message(
    bot: CQHttp,
    group_id: int,
    text: str,
    max_length: int = 2000,
    split_send: bool = True,
    split_segment_length: int = 500,
    split_delay: float = 0.5,
) -> dict:
    """向指定群发送文本消息，支持分段发送

    Args:
        bot: aiocqhttp bot实例
        group_id: 目标群号
        text: 消息内容
        max_length: 最大字符数，0表示不限制
        split_send: 是否分段发送长文本
        split_segment_length: 分段长度
        split_delay: 分段发送间隔（秒）

    Returns:
        发送结果dict
    """
    if max_length > 0 and len(text) > max_length:
        text = text[:max_length] + "\n...(消息过长已截断)"

    try:
        if split_send and len(text) > split_segment_length:
            segments = _split_text_segments(text, split_segment_length)
            last_result = None
            for i, seg in enumerate(segments):
                message = [{"type": "text", "data": {"text": seg}}]
                last_result = await bot.send_group_msg(group_id=group_id, message=message)
                if i < len(segments) - 1:
                    await asyncio.sleep(split_delay)
            logger.info(f"[CrossFlow] 跨群分段发送成功: group_id={group_id}, 共{len(segments)}段")
            return {"ok": True, "data": last_result, "segments": len(segments)}
        else:
            message = [{"type": "text", "data": {"text": text}}]
            result = await bot.send_group_msg(group_id=group_id, message=message)
            logger.info(f"[CrossFlow] 跨群发送成功: group_id={group_id}")
            return {"ok": True, "data": result}
    except Exception as e:
        logger.warning(f"[CrossFlow] 跨群发送失败: group_id={group_id}, error={e}")
        return {"ok": False, "error": str(e)}


async def send_private_message(
    bot: CQHttp,
    user_id: int,
    text: str,
    group_id: Optional[int] = None,
    max_length: int = 2000,
    split_send: bool = False,
    split_segment_length: int = 500,
    split_delay: float = 0.5,
) -> dict:
    """向指定用户发送私聊消息，支持分段发送

    Args:
        bot: aiocqhttp bot实例
        user_id: 目标用户QQ号
        text: 消息内容
        group_id: 可选，通过此群的临时会话通道发送（用于非好友）
        max_length: 最大字符数，0表示不限制
        split_send: 是否分段发送
        split_segment_length: 分段长度
        split_delay: 分段间隔（秒）

    Returns:
        发送结果dict
    """
    if max_length > 0 and len(text) > max_length:
        text = text[:max_length] + "\n...(消息过长已截断)"

    base_params = {"user_id": user_id}
    if group_id is not None:
        base_params["group_id"] = group_id

    try:
        if split_send and len(text) > split_segment_length:
            segments = _split_text_segments(text, split_segment_length)
            last_result = None
            for i, seg in enumerate(segments):
                message = [{"type": "text", "data": {"text": seg}}]
                last_result = await bot.send_private_msg(**base_params, message=message)
                if i < len(segments) - 1:
                    await asyncio.sleep(split_delay)
            channel = f"临时会话(via群{group_id})" if group_id else "私聊"
            logger.info(f"[CrossFlow] {channel}分段发送成功: user_id={user_id}, 共{len(segments)}段")
            return {"ok": True, "data": last_result, "channel": channel, "segments": len(segments)}
        else:
            message = [{"type": "text", "data": {"text": text}}]
            result = await bot.send_private_msg(**base_params, message=message)
            channel = f"临时会话(via群{group_id})" if group_id else "私聊"
            logger.info(f"[CrossFlow] {channel}发送成功: user_id={user_id}")
            return {"ok": True, "data": result, "channel": channel}
    except Exception as e:
        logger.warning(f"[CrossFlow] 私聊发送失败: user_id={user_id}, error={e}")
        return {"ok": False, "error": str(e)}


async def get_bot_group_list(bot: CQHttp) -> list[dict]:
    """获取机器人加入的所有群列表"""
    try:
        groups = await bot.get_group_list()
        return groups if isinstance(groups, list) else []
    except Exception as e:
        logger.warning(f"[CrossFlow] 获取群列表失败: {e}")
        return []


async def get_friend_list(bot: CQHttp) -> list[dict]:
    """获取机器人好友列表"""
    try:
        friends = await bot.get_friend_list()
        return friends if isinstance(friends, list) else []
    except Exception as e:
        logger.warning(f"[CrossFlow] 获取好友列表失败: {e}")
        return []


async def is_friend(bot: CQHttp, user_id: int) -> bool:
    """检查用户是否为好友"""
    friends = await get_friend_list(bot)
    return any(int(f.get("user_id", 0)) == user_id for f in friends)


async def find_common_group(bot: CQHttp, user_id: int) -> Optional[int]:
    """查找与目标用户共同所在的群组

    遍历机器人加入的所有群，检查目标用户是否在其中。
    找到第一个共同群就返回其group_id。

    Args:
        bot: aiocqhttp bot实例
        user_id: 目标用户QQ号

    Returns:
        共同群的group_id，如果没有共同群则返回None
    """
    groups = await get_bot_group_list(bot)
    if not groups:
        return None

    for group in groups:
        gid = int(group.get("group_id", 0))
        if gid <= 0:
            continue
        try:
            members = await bot.call_action(
                "get_group_member_list", group_id=gid
            )
            if not isinstance(members, list):
                continue
            for member in members:
                if int(member.get("user_id", 0)) == user_id:
                    logger.info(
                        f"[CrossFlow] 找到共同群: user_id={user_id}, "
                        f"group_id={gid}, group_name={group.get('group_name', '')}"
                    )
                    return gid
        except Exception as e:
            logger.debug(f"[CrossFlow] 检查群{gid}成员失败: {e}")
            continue

    logger.info(f"[CrossFlow] 未找到与 user_id={user_id} 的共同群")
    return None


async def smart_private_send(
    bot: CQHttp,
    user_id: int,
    text: str,
    enable_temp_session: bool = True,
    max_length: int = 2000,
    split_send: bool = False,
    split_segment_length: int = 500,
    split_delay: float = 0.5,
) -> dict:
    """智能私聊发送

    1. 先尝试直接私聊
    2. 如果失败且启用了临时会话，自动查找共同群组
    3. 通过共同群的临时会话通道重试

    Args:
        bot: aiocqhttp bot实例
        user_id: 目标用户QQ号
        text: 消息内容
        enable_temp_session: 是否启用临时会话降级
        max_length: 最大字符数

    Returns:
        发送结果dict
    """
    # 先检查是否是好友
    friend = await is_friend(bot, user_id)

    if friend:
        return await send_private_message(
            bot, user_id, text, max_length=max_length,
            split_send=split_send, split_segment_length=split_segment_length, split_delay=split_delay,
        )

    # 不是好友
    if not enable_temp_session:
        result = await send_private_message(
            bot, user_id, text, max_length=max_length,
            split_send=split_send, split_segment_length=split_segment_length, split_delay=split_delay,
        )
        if not result["ok"]:
            result["hint"] = "目标用户不是好友，且未启用临时会话功能"
        return result

    # 启用了临时会话，查找共同群
    common_gid = await find_common_group(bot, user_id)
    if common_gid:
        return await send_private_message(
            bot, user_id, text, group_id=common_gid, max_length=max_length,
            split_send=split_send, split_segment_length=split_segment_length, split_delay=split_delay,
        )
    else:
        result = await send_private_message(
            bot, user_id, text, max_length=max_length,
            split_send=split_send, split_segment_length=split_segment_length, split_delay=split_delay,
        )
        if not result["ok"]:
            result["hint"] = "目标用户不是好友，且未找到共同群组，无法发送临时会话"
        return result


def check_permission(
    sender_id: str,
    is_admin: bool,
    allowed_sender_ids: list[str],
) -> bool:
    """检查发送者是否有权限使用跨聊天功能

    Args:
        sender_id: 发送者QQ号
        is_admin: 是否为管理员
        allowed_sender_ids: 允许的发送者列表（空列表=不限制，所有人可用）

    Returns:
        是否有权限
    """
    # 管理员始终有权限
    if is_admin:
        return True
    # 如果白名单为空，表示不限制，所有人都可以用
    if not allowed_sender_ids:
        return True
    # 如果白名单不为空，检查发送者是否在白名单中
    if str(sender_id) in allowed_sender_ids:
        return True
    return False


def check_target_allowed(
    target_id: str,
    target_type: str,
    allowed_group_ids: list[str],
    allowed_user_ids: list[str],
) -> bool:
    """检查目标是否在白名单中

    Args:
        target_id: 目标ID
        target_type: "group" 或 "private"
        allowed_group_ids: 允许的群号列表（空=不限制）
        allowed_user_ids: 允许的用户列表（空=不限制）

    Returns:
        是否允许发送
    """
    if target_type == "group":
        if not allowed_group_ids:
            return True  # 空列表=不限制
        return str(target_id) in allowed_group_ids
    elif target_type == "private":
        if not allowed_user_ids:
            return True
        return str(target_id) in allowed_user_ids
    return False


def parse_target(text: str) -> Optional[dict]:
    """从用户输入中解析目标信息

    支持格式：
      /cf发群 <群号> <消息内容>
      /cf发私聊 <QQ号> <消息内容>
      /cf发临时 <QQ号> <消息内容>

    Args:
        text: 用户输入（已去除命令前缀的部分）

    Returns:
        解析结果dict或None
    """
    parts = text.strip().split(maxsplit=1)
    if len(parts) < 2:
        return None

    target_id_str = parts[0].strip()
    message = parts[1].strip() if len(parts) > 1 else ""

    if not target_id_str.isdigit():
        return None
    if not message:
        return None

    return {
        "target_id": int(target_id_str),
        "target_id_str": target_id_str,
        "message": message,
    }