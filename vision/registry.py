from dataclasses import dataclass, field
import numpy as np

@dataclass
class Person:
    person_id: str
    gallery: list[np.ndarray] = field(default_factory=list)
    protected: bool = False
    protection_source: str | None = None
    last_seen_ms: int = 0
    visible: bool = False
    last_sample_ms: int = -1000

class Registry:
    def __init__(self):
        self.people: dict[str, Person] = {}

    def create(self):
        person_id = f"p_{len(self.people)+1:03d}"
        person = Person(person_id)
        self.people[person_id] = person
        return person

    def protect(self, person_id, source):
        person = self.people[person_id]
        changed = not person.protected
        person.protected = True
        if changed:
            person.protection_source = source
        return changed

    def unprotect(self, person_id, source):
        person=self.people[person_id]
        changed=person.protected
        person.protected=False
        if changed: person.protection_source=None
        return changed

    @property
    def has_protected(self):
        return any(p.protected for p in self.people.values())
