"""Conservative PUCT search over candidate skill edits."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .evolver import SkillEvolver, SkillProposal
from .scorer import LLMScorer


@dataclass
class TreeNode:
    skill_name: str
    skill_content: str
    parent: "TreeNode | None"
    proposal: SkillProposal | None
    visit_count: int = 0
    total_q: float = 0.0
    children: list["TreeNode"] = field(default_factory=list)

    @property
    def q_value(self) -> float:
        return self.total_q / self.visit_count if self.visit_count else 0.0

    def puct_score(self, c_puct: float, parent_visits: int) -> float:
        prior = self.proposal.prior_confidence if self.proposal else 0.5
        exploration = c_puct * prior * math.sqrt(max(parent_visits, 1)) / (1 + self.visit_count)
        return self.q_value + exploration


class PUCTSearch:
    def __init__(
        self,
        evolver: SkillEvolver,
        scorer: LLMScorer,
        c_puct: float = 0.5,
        max_depth: int = 3,
        n_iterations: int = 20,
        proposals_per_node: int = 6,
    ):
        self.evolver = evolver
        self.scorer = scorer
        self.c_puct = c_puct
        self.max_depth = max_depth
        self.n_iterations = n_iterations
        self.k = proposals_per_node

    def search(
        self,
        skill_name: str,
        current_skill_content: str,
        failure_summary: str,
        trajectory_group: dict[str, Any],
    ) -> list[SkillProposal]:
        root = TreeNode(
            skill_name=skill_name,
            skill_content=current_skill_content,
            parent=None,
            proposal=None,
        )

        for _ in range(self.n_iterations):
            node = self._select(root)
            if self._get_depth(node) < self.max_depth:
                self._expand(node, failure_summary)
            children = node.children or ([node] if node.proposal else [])
            for child in children:
                q_value = self.scorer.batch_score(
                    evolved_skill_content=child.skill_content,
                    original_skill_content=current_skill_content,
                    skill_name=skill_name,
                    trajectory_group=trajectory_group,
                    top_k_failures=3,
                )
                self._backpropagate(child, q_value)

        leaves = self._collect_leaves(root)
        leaves.sort(key=lambda item: item.q_value, reverse=True)
        return [leaf.proposal for leaf in leaves[:3] if leaf.proposal]

    def _select(self, node: TreeNode) -> TreeNode:
        while node.children:
            node = max(
                node.children,
                key=lambda child: child.puct_score(self.c_puct, node.visit_count),
            )
        return node

    def _expand(self, node: TreeNode, failure_summary: str) -> None:
        if node.children:
            return
        proposals = self.evolver.generate_proposals(
            skill_name=node.skill_name,
            failure_summary=failure_summary,
            k=self.k,
        )
        for proposal in proposals:
            node.children.append(
                TreeNode(
                    skill_name=proposal.skill_name,
                    skill_content=proposal.new_content,
                    parent=node,
                    proposal=proposal,
                )
            )

    def _backpropagate(self, node: TreeNode, q_value: float) -> None:
        current: TreeNode | None = node
        while current is not None:
            current.visit_count += 1
            current.total_q += q_value
            current = current.parent

    def _get_depth(self, node: TreeNode) -> int:
        depth = 0
        current = node
        while current.parent is not None:
            depth += 1
            current = current.parent
        return depth

    def _collect_leaves(self, node: TreeNode) -> list[TreeNode]:
        if not node.children:
            return [node]
        leaves: list[TreeNode] = []
        for child in node.children:
            leaves.extend(self._collect_leaves(child))
        return leaves
