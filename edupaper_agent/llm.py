from __future__ import annotations

import re
from typing import Any

from openai import OpenAI

from .config import Settings
from .models import QueryMode


SYSTEM_PROMPT = """你是 EduPaper Agent，一名严谨的教育研究与学术文献助手。
你只能使用给定证据作答，不得把模型记忆当作事实来源。

规则：
1. 每个重要事实后使用证据编号，例如 [L1] 或 [A1]。
2. 证据不足时明确写出“现有证据不足”，不要推测精确数字、结论或因果关系。
3. qa 模式直接回答问题；review 模式按研究问题、理论框架、方法、样本、发现、局限组织；compare 模式使用对比维度组织。
4. 区分论文摘要中的作者主张与已被证实的事实。
5. 使用专业、简洁的中文。
"""


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client: OpenAI | None = None
        if settings.openai_api_key:
            kwargs: dict[str, str] = {"api_key": settings.openai_api_key}
            if settings.openai_base_url:
                kwargs["base_url"] = settings.openai_base_url
            self.client = OpenAI(**kwargs)

    @property
    def available(self) -> bool:
        return self.client is not None

    def generate(
        self,
        *,
        query: str,
        mode: QueryMode,
        evidence: list[dict[str, Any]],
    ) -> str:
        if not evidence:
            return "现有证据不足。请先上传相关 PDF，或允许系统检索 arXiv。"

        if self.client is None:
            return self._fallback_answer(query=query, mode=mode, evidence=evidence)

        blocks: list[str] = []
        used_chars = 0
        for item in evidence:
            block = (
                f"[{item['citation_id']}] 来源：{item['source']}；"
                f"页码：{item.get('page') or '-'}；章节：{item.get('section') or '-'}\n"
                f"{item['text']}"
            )
            if used_chars + len(block) > self.settings.max_context_chars:
                break
            blocks.append(block)
            used_chars += len(block)

        user_prompt = (
            f"回答模式：{mode.value}\n"
            f"用户问题：{query}\n\n"
            "可用证据：\n" + "\n\n".join(blocks)
        )
        response = self.client.chat.completions.create(
            model=self.settings.model_name,
            temperature=0.1,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content or "模型未生成有效回答。"

    @staticmethod
    def _fallback_answer(
        *, query: str, mode: QueryMode, evidence: list[dict[str, Any]]
    ) -> str:
        heading = {
            QueryMode.QA: "基于当前检索证据，相关内容如下：",
            QueryMode.REVIEW: "未配置模型 API，以下为可用于文献综述的证据摘要：",
            QueryMode.COMPARE: "未配置模型 API，以下为可用于比较的证据摘要：",
        }[mode]
        lines = [heading]
        for item in evidence[:6]:
            snippet = re.sub(r"\s+", " ", item["text"]).strip()[:320]
            lines.append(f"- [{item['citation_id']}] {snippet}")
        lines.append("\n提示：配置 OPENAI_API_KEY 后可生成结构化综合分析。")
        return "\n".join(lines)
