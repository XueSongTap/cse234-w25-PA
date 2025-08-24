from typing import Dict, List

import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import auto_diff as ad

def check_evaluator_output(
    evaluator: ad.Evaluator,
    input_values: Dict[ad.Node, torch.Tensor],
    expected_outputs: List[torch.Tensor],
) -> None:
    output_values = evaluator.run(input_values)
    assert len(output_values) == len(expected_outputs)
    for output_val, expected_val in zip(output_values, expected_outputs):
        torch.testing.assert_close(output_val, expected_val, atol=1e-4, rtol=1e-4)


def test_graph():
    x1 = ad.Variable("x1")
    x2 = ad.Variable("x2")
    x3 = ad.Variable("x3")
    trans_x2 = ad.transpose(x2, 1, 0)
    y = ad.matmul(x1, trans_x2) / 10 * x3
    x1_grad, x2_grad, x3_grad = ad.gradients(y, nodes=[x1, x2, x3])
    evaluator = ad.Evaluator(eval_nodes=[x1_grad, x2_grad, x3_grad])

    check_evaluator_output(
        evaluator,
        input_values={
            x1: torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]]),
            x2: torch.tensor([[2.8, 0.7, -0.1, 0.0], [0.6, 6.6, 3.2, 3.1]]),
            x3: torch.tensor([[2.71, 3.14], [3.87, -4.0]]),
        },
        expected_outputs=[
            torch.tensor(
                [[0.9472, 2.2621, 0.9777, 0.9734], [0.8436, -2.3691, -1.3187, -1.24]]
            ),
            torch.tensor(
                [[-0.1549, 0.542, -2.1091, 2.1211], [-0.434, 0.628, 2.477, -0.1724]]
            ),
            torch.tensor([[-0.145, 2.474], [0.142, -0.877]]),
        ],
    )


def test_gradient_of_gradient():
    x1 = ad.Variable(name="x1")
    x2 = ad.Variable(name="x2")
    y = x1 * x1 + x1 * x2

    grad_x1, grad_x2 = ad.gradients(y, [x1, x2])
    grad_x1_x1, grad_x1_x2 = ad.gradients(grad_x1, [x1, x2])
    grad_x2_x1, grad_x2_x2 = ad.gradients(grad_x2, [x1, x2])

    evaluator = ad.Evaluator(
        [y, grad_x1, grad_x2, grad_x1_x1, grad_x1_x2, grad_x2_x1, grad_x2_x2]
    )
    check_evaluator_output(
        evaluator,
        input_values={
            x1: torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]]),
            x2: torch.tensor([[2.8, 0.7, -0.1, 0.0], [0.6, 6.6, 3.2, 3.1]]),
        },
        expected_outputs=[
            torch.tensor([[-1.8, 5.4, 0.2, 11.56], [0.27, 0.0, 15.08, 19.22]]),
            torch.tensor([[0.8, 4.7, 0.9, 6.8], [1.2, 6.6, -8.4, 9.3]]),
            torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]]),
            2 * torch.ones((2, 4), dtype=torch.float32),
            1 * torch.ones((2, 4), dtype=torch.float32),
            1 * torch.ones((2, 4), dtype=torch.float32),
            torch.zeros((2, 4), dtype=torch.float32),
        ],
    )

def torch_expected_grads_for_graph(x1_val, x2_val, x3_val):
    x1_t = x1_val.clone().detach().requires_grad_(True)
    x2_t = x2_val.clone().detach().requires_grad_(True)
    x3_t = x3_val.clone().detach().requires_grad_(True)

    y_t = ((x1_t @ x2_t.t()) / 10) * x3_t   # 与现有 test_graph 同构（只是 * 而非 +）
    # 让标量 loss，便于反传；求和相当于全 ones 的外部梯度
    loss = y_t.sum()
    loss.backward()

    return x1_t.grad, x2_t.grad, x3_t.grad


def test_graph_backward_with_mean_keepdim_false():
    # y = mean( (x1 @ x2^T)/10 + x3, dim=1, keepdim=False )  -> shape (B,)
    # 用 torch autograd 生成期望梯度（通过把每个样本的标量和再求和）
    x1 = ad.Variable("x1")
    x2 = ad.Variable("x2")
    x3 = ad.Variable("x3")

    y = ad.mean(ad.matmul(x1, ad.transpose(x2, 1, 0)) / 10 + x3, dim=(1,), keepdim=False)
    # 为了有标量 loss，取 sum
    loss = ad.sum_op(y, dim=(0,), keepdim=False)

    x1_grad, x2_grad, x3_grad = ad.gradients(loss, [x1, x2, x3])
    evaluator = ad.Evaluator([x1_grad, x2_grad, x3_grad])

    B, T, H = 2, 3, 4
    x1_val = torch.randn(B, H)
    x2_val = torch.randn(T, H)
    x3_val = torch.randn(B, T)

    # Torch 期望：等价于 ((x1@x2^T)/10 + x3).mean(dim=1) 之后再对 batch 求和
    x1_t = x1_val.clone().detach().requires_grad_(True)
    x2_t = x2_val.clone().detach().requires_grad_(True)
    x3_t = x3_val.clone().detach().requires_grad_(True)
    y_t = ((x1_t @ x2_t.t()) / 10 + x3_t).mean(dim=1, keepdim=False)
    loss_t = y_t.sum()
    loss_t.backward()
    exp_x1, exp_x2, exp_x3 = x1_t.grad, x2_t.grad, x3_t.grad

    check_evaluator_output(
        evaluator,
        input_values={x1: x1_val, x2: x2_val, x3: x3_val},
        expected_outputs=[exp_x1, exp_x2, exp_x3],
    )


def test_graph_backward_shared_node_accumulation():
    # y = (x + x) * (x - 2) ; dy/dx = 2*(x - 2) + (x + x)*1 = 4x - 4
    x = ad.Variable("x")
    y = ad.mul(ad.add(x, x), ad.add_by_const(x, -2.0))
    loss = ad.sum_op(y, dim=(0, 1), keepdim=False)
    (x_grad,) = ad.gradients(loss, [x])
    evaluator = ad.Evaluator([x_grad])

    x_val = torch.tensor([[1.0, -2.0, 3.0]], dtype=torch.float32)
    expected = 4 * x_val - 4
    check_evaluator_output(evaluator, {x: x_val}, [expected])


def test_graph_backward_unused_variable_zero_grad():
    # y 仅依赖 x1，x2 的梯度应为 0
    x1 = ad.Variable("x1")
    x2 = ad.Variable("x2")
    y = ad.mul_by_const(x1, 3.0)
    loss = ad.sum_op(y, dim=(0, 1), keepdim=False)
    x1_grad, x2_grad = ad.gradients(loss, [x1, x2])
    evaluator = ad.Evaluator([x1_grad, x2_grad])

    x1_val = torch.randn(2, 3)
    x2_val = torch.randn(2, 3)
    expected_x1 = torch.full_like(x1_val, 3.0)  # d/dx1 (3*x1) = 3
    expected_x2 = torch.zeros_like(x2_val)      # 未参与计算
    check_evaluator_output(evaluator, {x1: x1_val, x2: x2_val}, [expected_x1, expected_x2])


def test_graph_backward_transpose_broadcast_matmul_chain():
    # 与现有 test_graph 相似，但再叠加一次 sum 让梯度可对拍
    x1 = ad.Variable("x1")
    x2 = ad.Variable("x2")
    x3 = ad.Variable("x3")
    y = ad.matmul(x1, ad.transpose(x2, 1, 0)) / 10 * x3
    loss = ad.sum_op(y, dim=(0, 1), keepdim=False)

    x1_grad, x2_grad, x3_grad = ad.gradients(loss, [x1, x2, x3])
    evaluator = ad.Evaluator([x1_grad, x2_grad, x3_grad])

    x1_val = torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]])
    x2_val = torch.tensor([[2.8, 0.7, -0.1, 0.0], [0.6, 6.6, 3.2, 3.1]])
    x3_val = torch.tensor([[2.71, 3.14], [3.87, -4.0]])

    # 用 torch 生成期望梯度
    x1_t = x1_val.clone().detach().requires_grad_(True)
    x2_t = x2_val.clone().detach().requires_grad_(True)
    x3_t = x3_val.clone().detach().requires_grad_(True)
    y_t = (x1_t @ x2_t.t()) / 10 * x3_t
    loss_t = y_t.sum()
    loss_t.backward()
    exp_x1, exp_x2, exp_x3 = x1_t.grad, x2_t.grad, x3_t.grad

    check_evaluator_output(
        evaluator,
        input_values={x1: x1_val, x2: x2_val, x3: x3_val},
        expected_outputs=[exp_x1, exp_x2, exp_x3],
    )

if __name__ == "__main__":
    test_graph()
    test_gradient_of_gradient()
    test_graph_backward_with_mean_keepdim_false()
    test_graph_backward_shared_node_accumulation()
    test_graph_backward_unused_variable_zero_grad()
    test_graph_backward_transpose_broadcast_matmul_chain()