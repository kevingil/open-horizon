//! Horizon scheduling: how many tool-call rounds a rollout gets.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ScalingMode {
    Linear,
    Step,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RoundScheduler {
    /// Constant horizon.
    Fixed { horizon: u32 },
    /// Ramps from `start` to `end` over `ramp_steps` training steps
    /// (AgentGym-style ScalingInter-RL curriculum).
    Scaling {
        start: u32,
        end: u32,
        ramp_steps: u32,
        mode: ScalingMode,
    },
}

impl Default for RoundScheduler {
    fn default() -> Self {
        RoundScheduler::Fixed { horizon: 6 }
    }
}

impl RoundScheduler {
    pub fn current_horizon(&self, training_step: u32) -> u32 {
        match *self {
            RoundScheduler::Fixed { horizon } => horizon,
            RoundScheduler::Scaling {
                start,
                end,
                ramp_steps,
                mode,
            } => {
                if training_step == 0 {
                    return start;
                }
                if training_step >= ramp_steps {
                    return end;
                }
                match mode {
                    ScalingMode::Step => {
                        if training_step * 2 < ramp_steps {
                            start
                        } else {
                            end
                        }
                    }
                    ScalingMode::Linear => {
                        let progress = training_step as f64 / ramp_steps as f64;
                        (start as f64 + (end as f64 - start as f64) * progress).round() as u32
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_ramp() {
        let s = RoundScheduler::Scaling {
            start: 4,
            end: 16,
            ramp_steps: 100,
            mode: ScalingMode::Linear,
        };
        assert_eq!(s.current_horizon(0), 4);
        assert_eq!(s.current_horizon(50), 10);
        assert_eq!(s.current_horizon(100), 16);
        assert_eq!(s.current_horizon(500), 16);
    }

    #[test]
    fn step_ramp() {
        let s = RoundScheduler::Scaling {
            start: 4,
            end: 16,
            ramp_steps: 100,
            mode: ScalingMode::Step,
        };
        assert_eq!(s.current_horizon(49), 4);
        assert_eq!(s.current_horizon(50), 16);
    }
}
