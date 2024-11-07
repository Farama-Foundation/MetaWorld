import pytest

from metaworld import MT1
from metaworld.policies import (
    SawyerAssemblyV3Policy,
    SawyerBasketballV3Policy,
    SawyerBinPickingV3Policy,
    SawyerBoxCloseV3Policy,
    SawyerButtonPressTopdownV3Policy,
    SawyerButtonPressTopdownWallV3Policy,
    SawyerButtonPressV3Policy,
    SawyerButtonPressWallV3Policy,
    SawyerCoffeeButtonV3Policy,
    SawyerCoffeePullV3Policy,
    SawyerCoffeePushV3Policy,
    SawyerDialTurnV3Policy,
    SawyerDisassembleV3Policy,
    SawyerDoorCloseV3Policy,
    SawyerDoorLockV3Policy,
    SawyerDoorOpenV3Policy,
    SawyerDoorUnlockV3Policy,
    SawyerDrawerCloseV3Policy,
    SawyerDrawerOpenV3Policy,
    SawyerFaucetCloseV3Policy,
    SawyerFaucetOpenV3Policy,
    SawyerHammerV3Policy,
    SawyerHandInsertV3Policy,
    SawyerHandlePressSideV3Policy,
    SawyerHandlePressV3Policy,
    SawyerHandlePullSideV3Policy,
    SawyerHandlePullV3Policy,
    SawyerLeverPullV3Policy,
    SawyerPegInsertionSideV3Policy,
    SawyerPegUnplugSideV3Policy,
    SawyerPickOutOfHoleV3Policy,
    SawyerPickPlaceV3Policy,
    SawyerPickPlaceWallV3Policy,
    SawyerPlateSlideBackSideV3Policy,
    SawyerPlateSlideBackV3Policy,
    SawyerPlateSlideSideV3Policy,
    SawyerPlateSlideV3Policy,
    SawyerPushBackV3Policy,
    SawyerPushV3Policy,
    SawyerPushWallV3Policy,
    SawyerReachV3Policy,
    SawyerReachWallV3Policy,
    SawyerShelfPlaceV3Policy,
    SawyerSoccerV3Policy,
    SawyerStickPullV3Policy,
    SawyerStickPushV3Policy,
    SawyerSweepIntoV3Policy,
    SawyerSweepV3Policy,
    SawyerWindowCloseV3Policy,
    SawyerWindowOpenV3Policy,
)

policies = dict(
    {
        "assembly-V3": SawyerAssemblyV3Policy,
        "basketball-V3": SawyerBasketballV3Policy,
        "bin-picking-V3": SawyerBinPickingV3Policy,
        "box-close-V3": SawyerBoxCloseV3Policy,
        "button-press-topdown-V3": SawyerButtonPressTopdownV3Policy,
        "button-press-topdown-wall-V3": SawyerButtonPressTopdownWallV3Policy,
        "button-press-V3": SawyerButtonPressV3Policy,
        "button-press-wall-V3": SawyerButtonPressWallV3Policy,
        "coffee-button-V3": SawyerCoffeeButtonV3Policy,
        "coffee-pull-V3": SawyerCoffeePullV3Policy,
        "coffee-push-V3": SawyerCoffeePushV3Policy,
        "dial-turn-V3": SawyerDialTurnV3Policy,
        "disassemble-V3": SawyerDisassembleV3Policy,
        "door-close-V3": SawyerDoorCloseV3Policy,
        "door-lock-V3": SawyerDoorLockV3Policy,
        "door-open-V3": SawyerDoorOpenV3Policy,
        "door-unlock-V3": SawyerDoorUnlockV3Policy,
        "drawer-close-V3": SawyerDrawerCloseV3Policy,
        "drawer-open-V3": SawyerDrawerOpenV3Policy,
        "faucet-close-V3": SawyerFaucetCloseV3Policy,
        "faucet-open-V3": SawyerFaucetOpenV3Policy,
        "hammer-V3": SawyerHammerV3Policy,
        "hand-insert-V3": SawyerHandInsertV3Policy,
        "handle-press-side-V3": SawyerHandlePressSideV3Policy,
        "handle-press-V3": SawyerHandlePressV3Policy,
        "handle-pull-V3": SawyerHandlePullV3Policy,
        "handle-pull-side-V3": SawyerHandlePullSideV3Policy,
        "peg-insert-side-V3": SawyerPegInsertionSideV3Policy,
        "lever-pull-V3": SawyerLeverPullV3Policy,
        "peg-unplug-side-V3": SawyerPegUnplugSideV3Policy,
        "pick-out-of-hole-V3": SawyerPickOutOfHoleV3Policy,
        "pick-place-V3": SawyerPickPlaceV3Policy,
        "pick-place-wall-V3": SawyerPickPlaceWallV3Policy,
        "plate-slide-back-side-V3": SawyerPlateSlideBackSideV3Policy,
        "plate-slide-back-V3": SawyerPlateSlideBackV3Policy,
        "plate-slide-side-V3": SawyerPlateSlideSideV3Policy,
        "plate-slide-V3": SawyerPlateSlideV3Policy,
        "reach-V3": SawyerReachV3Policy,
        "reach-wall-V3": SawyerReachWallV3Policy,
        "push-back-V3": SawyerPushBackV3Policy,
        "push-V3": SawyerPushV3Policy,
        "push-wall-V3": SawyerPushWallV3Policy,
        "shelf-place-V3": SawyerShelfPlaceV3Policy,
        "soccer-V3": SawyerSoccerV3Policy,
        "stick-pull-V3": SawyerStickPullV3Policy,
        "stick-push-V3": SawyerStickPushV3Policy,
        "sweep-into-V3": SawyerSweepIntoV3Policy,
        "sweep-V3": SawyerSweepV3Policy,
        "window-close-V3": SawyerWindowCloseV3Policy,
        "window-open-V3": SawyerWindowOpenV3Policy,
    }
)


@pytest.mark.parametrize("env_name", MT1.ENV_NAMES)
def test_policy(env_name):
    mt1 = MT1(env_name)
    env = mt1.train_classes[env_name]()
    p = policies[env_name]()
    completed = 0
    for task in mt1.train_tasks:
        env.set_task(task)
        obs, info = env.reset()
        done = False
        count = 0
        while count < 500 and not done:
            count += 1
            a = p.get_action(obs)
            next_obs, _, trunc, termn, info = env.step(a)
            done = trunc or termn
            obs = next_obs
            if int(info["success"]) == 1:
                completed += 1
                break
    print(float(completed) / 50)
    assert (float(completed) / 50) > 0.80
