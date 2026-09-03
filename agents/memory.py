"""
memory.py

Agent Memory

Stores authentication history and agent context
for intelligent decision-making.
"""

from datetime import datetime


class AgentMemory:

    def __init__(self):

        self.memory = {}

    # ==========================================================
    # Store Value
    # ==========================================================

    def set(self, key, value):

        self.memory[key] = value

    # ==========================================================
    # Retrieve Value
    # ==========================================================

    def get(self, key, default=None):

        return self.memory.get(key, default)

    # ==========================================================
    # Remove Value
    # ==========================================================

    def remove(self, key):

        if key in self.memory:
            del self.memory[key]

    # ==========================================================
    # Clear Memory
    # ==========================================================

    def clear(self):

        self.memory.clear()

    # ==========================================================
    # Complete Memory
    # ==========================================================

    def all(self):

        return self.memory

    # ==========================================================
    # Authentication History
    # ==========================================================

    def add_authentication(
        self,
        username,
        confidence,
        risk,
        decision,
    ):

        history = self.memory.get(
            "authentication_history",
            []
        )

        history.append({

            "username": username,

            "confidence": confidence,

            "risk": risk,

            "decision": decision,

            "time": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )

        })

        self.memory["authentication_history"] = history

    # ==========================================================
    # Success Counter
    # ==========================================================

    def increment_success(self):

        self.memory["success_count"] = \
            self.memory.get(
                "success_count",
                0
            ) + 1

    # ==========================================================
    # Failure Counter
    # ==========================================================

    def increment_failure(self):

        self.memory["failure_count"] = \
            self.memory.get(
                "failure_count",
                0
            ) + 1

    # ==========================================================
    # Trust Score
    # ==========================================================

    def calculate_trust_score(self):

        success = self.memory.get(
            "success_count",
            0
        )

        failure = self.memory.get(
            "failure_count",
            0
        )

        total = success + failure

        if total == 0:

            return 100.0

        return round(
            (success / total) * 100,
            2
        )

    # ==========================================================
    # Last Authentication
    # ==========================================================

    def last_authentication(self):

        history = self.memory.get(
            "authentication_history",
            []
        )

        if len(history) == 0:

            return None

        return history[-1]

    # ==========================================================
    # Summary
    # ==========================================================

    def summary(self):

        return {

            "success_count":
                self.memory.get(
                    "success_count",
                    0
                ),

            "failure_count":
                self.memory.get(
                    "failure_count",
                    0
                ),

            "trust_score":
                self.calculate_trust_score(),

            "last_authentication":
                self.last_authentication()

        }


# ==========================================================
# Testing
# ==========================================================

if __name__ == "__main__":

    memory = AgentMemory()

    memory.increment_success()

    memory.increment_success()

    memory.increment_failure()

    memory.add_authentication(

        username="doctor",

        confidence=0.98,

        risk="LOW",

        decision="GRANT_ACCESS"

    )

    print(memory.summary())