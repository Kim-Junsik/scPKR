"""How condition names are written, in one place.

Norman spells a double 'AHR+FEV', a single 'AHR+ctrl' and the control 'ctrl'.
None of that is universal - combosciplex uses drug names, and other datasets use
'control' or 'DMSO' - so the convention is configuration, not a constant.

It used to be a module-level constant in five different files while
`config.data.control_label` existed and was read in exactly one of them. A dataset
that named its control anything else would have raised a KeyError somewhere deep
in evaluation, which is the kind of failure that looks like a bug in the model.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConditionNaming:
    control: str = "ctrl"
    separator: str = "+"

    @classmethod
    def from_config(cls, config: dict) -> "ConditionNaming":
        data = config.get("data", {})
        return cls(control=data.get("control_label", "ctrl"),
                   separator=data.get("condition_separator", "+"))

    # ---------------------------------------------------------------- parsing
    def genes(self, condition: str) -> list[str]:
        """Perturbations named by a condition. Control contributes none."""
        return [g for g in condition.split(self.separator) if g != self.control]

    def is_control(self, condition: str) -> bool:
        return not self.genes(condition)

    def arity(self, condition: str) -> int:
        return len(self.genes(condition))

    def single(self, gene: str) -> str:
        """The canonical single-perturbation condition for `gene`."""
        return f"{gene}{self.separator}{self.control}"

    def single_forms(self, gene: str) -> list[str]:
        """Both spellings, because datasets disagree on which side control goes.

        Norman writes 'AHR+ctrl'; combosciplex writes 'control+Dacinostat'. Asking
        for only the canonical one made every single lookup miss on combosciplex,
        which reads as "not computable" rather than "wrong spelling" - the harness
        would have skipped every additive baseline and reported nothing wrong.
        """
        return [f"{gene}{self.separator}{self.control}",
                f"{self.control}{self.separator}{gene}"]

    def is_single(self, condition: str) -> bool:
        return self.arity(condition) == 1

    def is_double(self, condition: str) -> bool:
        return self.arity(condition) == 2


DEFAULT = ConditionNaming()
