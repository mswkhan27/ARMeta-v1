import time

class TestManager:
    def __init__(self, target_cov=85, plateau_window=5, max_requests=1000, max_minutes=30):
        self.target_cov = target_cov
        self.plateau_window = plateau_window
        self.max_requests = max_requests
        self.max_minutes = max_minutes
        self.start_ts = time.time()
        self.coverage_history = []
        self.op_hits_history = []
        self.request_count = 0
        self.server_crashes = 0
        # Optional: relay storage for multi-agent pipelines (agent -> manager -> next agent).
        # Not used unless explicitly called; does not affect existing behavior/output.
        self._agent_outputs = {}

        # Optional: loop control state (GUI can delegate loop gating to the manager).
        # Not used unless explicitly called; does not affect existing behavior/output.
        self.iteration = 0
        self._stopped = False
        self._stop_reason = None

    def update_metrics(self, coverage, unique_ops, requests, crashes):
        self.coverage_history.append(coverage)
        self.op_hits_history.append(unique_ops)
        self.request_count = requests
        self.server_crashes += crashes

    def should_stop(self):
        # 1. Coverage reached
        if self.coverage_history and self.coverage_history[-1] >= self.target_cov:
            return True, "Reached target operation coverage."

        # 2. Plateau detection
        if self.plateau_window > 0 and len(self.op_hits_history) >= self.plateau_window:
            gained = self.op_hits_history[-1] - self.op_hits_history[-self.plateau_window]
            if gained < 1:
                return True, f"No new operations in {self.plateau_window} iterations."

        # 3. Request budget
        if self.request_count >= self.max_requests:
            return True, f"Request limit reached ({self.max_requests})."

        # 4. Time budget
        if (time.time() - self.start_ts) / 60.0 >= self.max_minutes:
            return True, f"Time budget exceeded ({self.max_minutes} min)."

        # 5. Crash rate
        if self.request_count and (self.server_crashes / self.request_count) >= 0.05:
            return True, "Crash rate ≥ 5%."

        return False, ""

    # ---- Optional relay helpers (agent -> manager -> next agent) ----
    def record_agent_output(self, agent_name: str, output):
        """Store an agent's latest output for the manager to pass onwards."""
        self._agent_outputs[str(agent_name)] = output
        return output

    def get_agent_output(self, agent_name: str, default=None):
        """Fetch the latest stored output for an agent."""
        return self._agent_outputs.get(str(agent_name), default)

    def clear_agent_outputs(self):
        """Clear all stored agent outputs."""
        self._agent_outputs.clear()

    # ---- Optional loop control helpers (manager controls the loop) ----
    def start_loop(self, start_iteration: int = 0):
        self.iteration = int(start_iteration or 0)
        self._stopped = False
        self._stop_reason = None

    def continue_loop(self) -> bool:
        return not self._stopped

    def stop(self, reason=None):
        self._stopped = True
        self._stop_reason = reason
        return reason

    @property
    def stop_reason(self):
        return self._stop_reason

    def next_iteration(self) -> int:
        self.iteration += 1
        return self.iteration


# Backwards compatibility: keep old name without changing behavior.
Coordinator = TestManager
