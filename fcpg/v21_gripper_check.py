"""One empty-jaw response check at HOME, using the existing IO owner only."""
import numpy as np

from .v21_hardware_bridge import completion_hold_record


def empty_gripper_follow_check(owner, home):
    """Five targets, original q5, then read-only tail; restore opening separately.

    This is an explicit manual diagnostic, not an Actor transaction. The caller
    must verify empty jaws visually and clear the STOP latch once when accepting
    the request. This function never clears it and never disables torque.
    """
    bridge = owner.bridge
    if not bridge.lock.acquire(blocking=False):
        raise RuntimeError('another hardware transaction is active')
    result = dict(kind='empty-gripper-follow-check', model_transition=False,
                  completed=False, amplitude_position=6., phases=[])
    owner.gripper_check_record = result
    touched = False
    try:
        if owner.runtime.pending is not None or owner.runtime.executing:
            raise RuntimeError('stop the model transaction before a manual diagnostic')
        source,_ = bridge._read()
        home = np.asarray(home,np.float32)
        if home.shape != (6,) or not np.isfinite(home).all():
            raise ValueError('fixed HOME must be finite [6]')
        if np.max(np.abs(source-home)) > owner.io.settle_position_tolerance:
            raise RuntimeError('empty-gripper check requires the established HOME pose')
        if source[5] < 20.:
            raise RuntimeError('empty-gripper check requires an open starting position')
        opening = float(source[5])
        for label, final_opening in (('close',opening-6.),('restore',opening)):
            initial,_ = bridge._read()
            commands = np.repeat(initial[None],5,axis=0)
            commands[-1,5] = final_opening
            owner._physical_check(initial,commands)
            q0,epoch = bridge._read()
            if np.max(np.abs(q0-initial)) > owner.io.settle_step_tolerance:
                raise RuntimeError('source pose changed before the diagnostic')
            phase = dict(name=label, source_time_s=epoch, commands_position=[], command_times_s=[],
                         measured_q_path=[q0.tolist()], readback_times_s=[epoch],
                         tail_positions=[],tail_times_s=[])
            result['phases'].append(phase)
            previous=q0.copy()
            for i, command in enumerate(commands):
                bridge._boundary(epoch+i*.1)
                if i:
                    q,stamp=bridge._read()
                    phase['measured_q_path'].append(q.tolist());phase['readback_times_s'].append(stamp)
                bridge._boundary(epoch+i*.1)
                touched=True
                sent=bridge.clock()
                actual=np.asarray(bridge.write_q(command,previous),np.float32)
                phase['commands_position'].append(actual.tolist());phase['command_times_s'].append(sent)
                if actual.shape!=(6,) or not np.allclose(actual,command,atol=1e-4,rtol=0):
                    raise RuntimeError('transport changed the diagnostic target')
                bridge._boundary(epoch+i*.1)
                previous=actual.copy()
            bridge._boundary(epoch+.5)
            q5,stamp=bridge._read()
            bridge._boundary(epoch+.5)
            phase['measured_q_path'].append(q5.tolist());phase['readback_times_s'].append(stamp)
            # The original 0.5s sample remains intact. No additional command
            # is issued while observing the same goal for at most 0.4s more.
            for offset in (.55,.60,.65,.70,.75,.80,.85,.90):
                bridge._boundary(epoch+offset)
                q,stamp=bridge._read()
                bridge._boundary(epoch+offset)
                phase['tail_positions'].append(q.tolist());phase['tail_times_s'].append(stamp)
            started=bridge.clock()
            held=bridge.halt()
            phase['completion_hold']=completion_hold_record(held,started,bridge.clock())
        result['completed']=True
        return result
    except BaseException as exc:
        result['error']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        owner.stop_event.set();owner.runtime.stop()
        if touched: owner.runtime.state=None  # manual motion is not invented model history
        try:
            if not result['completed']: bridge.halt()
        finally: bridge.lock.release()
