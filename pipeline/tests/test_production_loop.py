from anchor.agent.production_loop import evaluate_production_candidate


def test_surrogate_physics_cannot_be_called_production_ready():
    result = evaluate_production_candidate(
        {"technical_pass": True},
        {"fit": {"joint_rmse_rad": 0.01}},
        {
            "exact_embodiment_mapping": False,
            "metrics": {"joint_tracking_rmse_rad": 0.02, "upright_fraction": 1.0, "joint_limit_violation_count": 0},
        },
        {"grasp_success_rate": 0.9, "minimum_hold_s": 6.0},
    )
    assert result["production_release_ready"] is False
    assert result["next_action"] == "install_exact_robot_collision_asset_and_actuator_model"


def test_complete_evidence_advances_to_supervised_review():
    result = evaluate_production_candidate(
        {"technical_pass": True},
        {"fit": {"joint_rmse_rad": 0.01}},
        {
            "exact_embodiment_mapping": True,
            "metrics": {"joint_tracking_rmse_rad": 0.02, "upright_fraction": 1.0, "joint_limit_violation_count": 0},
        },
        {"grasp_success_rate": 0.8, "minimum_hold_s": 5.0},
    )
    assert result["production_release_ready"] is True
    assert result["physical_robot_execution_authorized"] is False

