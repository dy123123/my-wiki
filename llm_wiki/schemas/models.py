"""Pydantic models for structured LLM outputs."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class EntityRef(BaseModel):
    name: str = Field(description="Full canonical name of the entity")
    type: Literal["person", "organization", "product", "place", "event", "other"] = Field(
        description="Entity type"
    )
    description: str = Field(description="One-sentence description of the entity")
    mentions: list[str] = Field(
        default_factory=list,
        description="Relevant quotes or context from the source document",
    )


class ConceptRef(BaseModel):
    name: str = Field(description="Name of the concept or technical term")
    description: str = Field(description="Plain-language definition in context")
    related_concepts: list[str] = Field(
        default_factory=list,
        description="Names of related concepts",
    )


class TopicRef(BaseModel):
    name: str = Field(description="Topic or theme name")
    description: str = Field(description="How the source document relates to this topic")


class SourceAnalysis(BaseModel):
    """Structured analysis of a source document."""

    title: str = Field(description="Document title (inferred if not explicit)")
    summary: str = Field(description="2-3 sentence summary of the document")
    key_points: list[str] = Field(
        description="Key takeaways or findings, each as a concise sentence"
    )
    entities: list[EntityRef] = Field(
        default_factory=list,
        description="Named entities (people, orgs, products, places, events) mentioned",
    )
    concepts: list[ConceptRef] = Field(
        default_factory=list,
        description="Key concepts and technical terms",
    )
    topics: list[TopicRef] = Field(
        default_factory=list,
        description="High-level topics covered",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Short lowercase tags for classification",
    )
    date_published: Optional[str] = Field(
        default=None,
        description="Publication date in YYYY-MM-DD format, or null if unknown",
    )
    authors: list[str] = Field(
        default_factory=list,
        description="Author names",
    )


class WikiPageResult(BaseModel):
    """Generated wiki page content."""

    content: str = Field(description="Full markdown content for the wiki page")


class AnswerResult(BaseModel):
    """LLM answer to a user question."""

    answer: str = Field(description="Direct answer to the question")
    reasoning: str = Field(description="Explanation and supporting details")
    citations: list[str] = Field(
        default_factory=list,
        description="List of wiki page paths consulted",
    )
    confidence: Literal["high", "medium", "low"] = Field(
        default="medium",
        description="Confidence level based on available wiki content",
    )
    gaps: list[str] = Field(
        default_factory=list,
        description="Information gaps: what would improve the answer if added to the wiki",
    )


class LintIssue(BaseModel):
    """A single lint issue found in the wiki."""

    severity: Literal["error", "warning", "info"]
    category: Literal["orphan", "dead_link", "duplicate", "no_citation", "empty_page", "missing_section"]
    file: str = Field(description="Relative path to the affected file")
    message: str
    fix_hint: Optional[str] = None


class LintReport(BaseModel):
    issues: list[LintIssue] = Field(default_factory=list)

    @property
    def errors(self) -> list[LintIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[LintIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    def summary(self) -> str:
        e = len(self.errors)
        w = len(self.warnings)
        i = len([x for x in self.issues if x.severity == "info"])
        return f"{e} error(s), {w} warning(s), {i} info"
