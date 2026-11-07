"""Section 17.80: checks that keep their answer on the GPU until it is read (fcpg/v21_checks.py)."""
import pytest
import torch

from fcpg.prod_v20_policy import PolicyContractError, project_joint_bounds
from fcpg.v21_checks import Pending, deferred, deferring, fail_if, later
from fcpg.v21_memory import finite, finite_where
from fcpg.v21_models import CommandTrace

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a GPU')


def test_a_failed_check_raises_at_once_outside_deferred():
    fail_if(torch.tensor(False), 'never')
    with pytest.raises(ValueError, match='at once'):
        fail_if(torch.tensor(True), 'at once')
    with pytest.raises(PolicyContractError, match='typed'):
        fail_if(torch.tensor(True), 'typed', PolicyContractError)
    seen = []
    later(torch.tensor([1., 2.5]), seen.append)
    assert seen == [[1., 2.5]] and not deferring()


def test_finite_where_is_finite_on_the_selected_values():
    value = torch.tensor([[1., float('nan')], [2., 3.]])
    finite_where('x', value, torch.tensor([[True, False], [True, True]]))      # the NaN is not selected
    with pytest.raises(ValueError, match='x: expected finite floating values'):
        finite_where('x', value, torch.tensor([[True, True], [False, False]]))
    with pytest.raises(ValueError, match='expected finite floating values'):
        finite_where('x', torch.ones(2, 2, dtype=torch.long), torch.ones(2, 2, dtype=torch.bool))
    finite('x', value[torch.tensor([[True, False], [True, True]])])


def test_command_trace_checks_raise_as_before():
    def trace(positions=None, offsets=None, valid=None):
        valid = torch.tensor([[True, True, False]]) if valid is None else valid
        positions = torch.zeros(1, 3, 6) if positions is None else positions
        offsets = torch.tensor([[0., .1, -5.]], dtype=torch.float64) if offsets is None else offsets
        return CommandTrace(positions, offsets, valid, 'recorded', True)

    trace().validate(1)                                       # the padding may hold anything
    bad = torch.zeros(1, 3, 6)
    bad[0, 1, 2] = float('inf')
    with pytest.raises(ValueError, match='valid commands: expected finite'):
        trace(positions=bad).validate(1)
    with pytest.raises(ValueError, match='negative command offset'):
        trace(offsets=torch.tensor([[-.1, .1, 0.]], dtype=torch.float64)).validate(1)
    with pytest.raises(ValueError, match='command times must increase strictly'):
        trace(offsets=torch.tensor([[.2, .1, 0.]], dtype=torch.float64)).validate(1)
    with pytest.raises(ValueError, match='nonempty valid prefix'):
        trace(valid=torch.tensor([[False, True, False]])).validate(1)
    with pytest.raises(ValueError, match='command offsets: timestamps must be float64 seconds'):
        trace(offsets=torch.tensor([[0., .1, 0.]])).validate(1)


def test_joint_bounds_check_their_inputs_as_before():
    lower, upper = torch.zeros(6), torch.ones(6)
    assert torch.equal(project_joint_bounds(torch.full((2, 6), 2.), lower, upper), torch.ones(2, 6))
    with pytest.raises(PolicyContractError, match='invalid projection bounds'):
        project_joint_bounds(torch.zeros(2, 6), upper, lower)
    with pytest.raises(PolicyContractError, match='commands must be finite'):
        project_joint_bounds(torch.full((2, 6), float('nan')), lower, upper)


@cuda
def test_deferred_checks_are_read_once_and_raise_the_first_failure_in_order():
    device = torch.device('cuda')
    pending, seen = Pending(), []
    with deferred(pending):
        assert deferring()
        fail_if(torch.zeros((), dtype=torch.bool, device=device), 'fine')
        later(torch.tensor([3., 4.], device=device), seen.append)
        fail_if(torch.ones((), dtype=torch.bool, device=device), 'first', RuntimeError)
        fail_if(torch.ones((), dtype=torch.bool, device=device), 'second')
        with pytest.raises(ValueError, match='a CPU flag'):
            fail_if(torch.tensor(True), 'a CPU flag still raises at once')
    assert not deferring() and not seen
    pending.start(torch.tensor(7., device=device))
    with pytest.raises(RuntimeError, match='first'):
        pending.finish((1,))

    pending, seen = Pending(), []
    with deferred(pending):
        fail_if(torch.zeros((), dtype=torch.bool, device=device), 'fine')
        later(torch.tensor([3., 4.], device=device), seen.append)
    pending.start(torch.tensor([7., 8.], device=device), torch.tensor(9, device=device))
    assert pending.finish((2, 1)) == [[7., 8.], [9.]] and seen == [[3., 4.]]


@cuda
def test_a_deferred_check_does_not_wait_for_the_gpu():
    device = torch.device('cuda')
    torch.cuda.set_sync_debug_mode('error')
    try:
        pending = Pending()
        with deferred(pending):
            value = torch.randn(64, 6, device=device)
            finite('q', value, (64, 6))
            finite_where('q', value, value[:, 0] > 0)
            later(value.sum(0), lambda values: None)
        pending.start(value.sum())
    finally:
        torch.cuda.set_sync_debug_mode('default')
    pending.finish((1,))
