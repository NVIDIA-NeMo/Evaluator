"""Tests for StepEnvironment multi-turn contract."""
import pytest
from nemo_evaluator.environments.base import StepEnvironment, Observation


class CountToFive(StepEnvironment):
    """Trivial multi-turn env: agent must say numbers 1-5 in order."""
    name = "count_to_five"

    def __init__(self):
        self._problems = [{"target": 5}]
        self._current = 0
        self._expected_next = 1

    def __len__(self):
        return len(self._problems)

    def reset(self, idx):
        self._current = 0
        self._expected_next = 1
        return Observation(content="Count from 1 to 5, one number at a time.")

    def step(self, action):
        try:
            num = int(action.strip())
        except ValueError:
            return Observation(content="That's not a number. Try again.", reward=0.0)

        if num == self._expected_next:
            self._expected_next += 1
            done = self._expected_next > 5
            return Observation(
                content="Correct!" if not done else "Done!",
                done=done,
                reward=1.0 if done else 0.0,
            )
        return Observation(content=f"Expected {self._expected_next}, got {num}.", reward=0.0)

    @property
    def max_steps(self):
        return 10


class TestStepEnvironment:
    def test_successful_episode(self):
        env = CountToFive()
        obs = env.reset(0)
        assert not obs.done
        assert "Count" in obs.content

        for i in range(1, 5):
            obs = env.step(str(i))
            assert not obs.done
            assert obs.reward == 0.0

        obs = env.step("5")
        assert obs.done
        assert obs.reward == 1.0

    def test_wrong_number(self):
        env = CountToFive()
        env.reset(0)
        obs = env.step("3")
        assert not obs.done
        assert "Expected 1" in obs.content

    def test_max_steps(self):
        env = CountToFive()
        assert env.max_steps == 10

    def test_len(self):
        env = CountToFive()
        assert len(env) == 1

    def test_observation_tools(self):
        obs = Observation(
            content="use the calculator",
            tools=[{"name": "calc", "schema": {"input": "str"}}],
        )
        assert obs.tools[0]["name"] == "calc"
