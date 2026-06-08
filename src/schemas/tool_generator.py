from pydantic import BaseModel, Field


class SkillTest(BaseModel):
    name: str
    scenario: str
    task: str = ""
    source_data: str = ""
    should_activate: bool
    expected_behavior: str
    expected_answer: str = ""
    expected_intermediate: list[str] = Field(default_factory=list)
    must_check: list[str] = Field(default_factory=list)
    must_reject: list[str] = Field(default_factory=list)
    forbidden_outputs: list[str] = Field(default_factory=list)
    regression_guard: str = ""


class ToolGeneratorResponse(BaseModel):
    generated_skill: str
    reasoning: str
    skill_path: str | None = None
    skill_markdown: str | None = None
    skill_tests: list[SkillTest] | None = None
