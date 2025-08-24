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
        print(repr(output_val))
        torch.testing.assert_close(output_val, expected_val, atol=1e-4, rtol=1e-4)


def test_mul():
    x1 = ad.Variable("x1")
    x2 = ad.Variable("x2")
    y = ad.mul(x1, x2)
    y_grad = ad.Variable("y_grad")
    x1_grad, x2_grad = y.op.gradient(y, y_grad)
    evaluator = ad.Evaluator(eval_nodes=[x1_grad, x2_grad])

    check_evaluator_output(
        evaluator,
        input_values={
            x1: torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]]),
            x2: torch.tensor([[2.8, 0.7, -0.1, 0.0], [0.6, 6.6, 3.2, 3.1]]),
            y_grad: torch.tensor([[-0.4, 0.5, -5.0, 34.0], [-0.0375, 0.0, 2.32, -3.1]]),
        },
        expected_outputs=[
            torch.tensor([[-1.12, 0.35, 0.5, 0.0], [-0.0225, 0.0, 7.424, -9.61]]),
            torch.tensor([[0.4, 1.0, -2.5, 115.6], [-0.01125, 0.0, -13.456, -9.61]]),
        ],
    )


def test_div():
    x1 = ad.Variable("x1")
    x2 = ad.Variable("x2")
    y = ad.div(x1, x2)
    y_grad = ad.Variable("y_grad")
    x1_grad, x2_grad = y.op.gradient(y, y_grad)
    evaluator = ad.Evaluator(eval_nodes=[x1_grad, x2_grad])

    check_evaluator_output(
        evaluator,
        input_values={
            x1: torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]]),
            x2: torch.tensor([[2.5, 4.0, -0.1, 0.1], [-8.0, 5.0, -2.5, -1.0]]),
            y_grad: torch.ones((2, 4), dtype=torch.float32),
        },
        expected_outputs=[
            torch.tensor([[0.4, 0.25, -10.0, 10.0], [-0.125, 0.2, -0.4, -1.0]]),
            torch.tensor([[0.16, -0.125, -50.0, -340.0], [-0.0046875, 0, 0.928, -3.1]]),
        ],
    )


def test_div_by_const():
    x1 = ad.Variable("x1")
    y = ad.div_by_const(x1, 5.0)
    evaluator = ad.Evaluator(eval_nodes=[y])

    check_evaluator_output(
        evaluator,
        input_values={x1: torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]])},
        expected_outputs=[torch.tensor([[-0.2, 0.4, 0.1, 0.68], [0.06, 0.0, -1.16, 0.62]])],
    )

def test_matmul():
    x1 = ad.Variable("x1")
    x2 = ad.Variable("x2")
    y = ad.matmul(x1, x2)
    y_grad = ad.Variable("y_grad")
    x1_grad, x2_grad = y.op.gradient(y, y_grad)
    evaluator = ad.Evaluator(eval_nodes=[x1_grad, x2_grad])

    x1_val = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    x2_val = torch.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
    y_grad_val = torch.ones((3, 3), dtype=torch.float32)
    x1_grad_expected = torch.tensor([[24.0, 33.0], [24.0, 33.0], [24.0, 33.0]])
    x2_grad_expected = torch.tensor([[9.0, 9.0, 9.0], [12.0, 12.0, 12.0]])

    check_evaluator_output(
        evaluator,
        input_values={
            x1: x1_val,
            x2: x2_val,
            y_grad: y_grad_val,
        },
        expected_outputs=[x1_grad_expected, x2_grad_expected],
    )

def test_matmul_3d():
    x1 = ad.Variable("x1")
    x2 = ad.Variable("x2")
    y = ad.matmul(x1, x2)
    y_grad = ad.Variable("y_grad")
    x1_grad, x2_grad = y.op.gradient(y, y_grad)
    evaluator = ad.Evaluator(eval_nodes=[x1_grad, x2_grad])

    x1_val = torch.tensor([[[1.0, 2.0, 3.0],
                           [4.0, 5.0, 6.0],
                           [7.0, 8.0, 9.0]],
                          [[9.0, 8.0, 7.0],
                           [6.0, 5.0, 4.0],
                           [3.0, 2.0, 1.0]]])
    
    x2_val = torch.tensor([[[1.0, 2.0, 3.0],
                           [4.0, 5.0, 6.0],
                           [7.0, 8.0, 9.0]],
                          [[9.0, 8.0, 7.0],
                           [6.0, 5.0, 4.0],
                           [3.0, 2.0, 1.0]]])

    y_grad_val = torch.ones((2, 3, 3), dtype=torch.float32)

    x1_grad_expected = torch.tensor([[[6.0, 15.0, 24.0],
                                    [6.0, 15.0, 24.0],
                                    [6.0, 15.0, 24.0]],
                                   [[24.0, 15.0, 6.0],
                                    [24.0, 15.0, 6.0],
                                    [24.0, 15.0, 6.0]]])

    x2_grad_expected = torch.tensor([[[12.0, 12.0, 12.0],
                                    [15.0, 15.0, 15.0],
                                    [18.0, 18.0, 18.0]],
                                   [[18.0, 18.0, 18.0],
                                    [15.0, 15.0, 15.0],
                                    [12.0, 12.0, 12.0]]])

    check_evaluator_output(
        evaluator,
        input_values={
            x1: x1_val,
            x2: x2_val,
            y_grad: y_grad_val,
        },
        expected_outputs=[x1_grad_expected, x2_grad_expected],
    )


def test_layernorm():
    x = ad.Variable("x")
    y = ad.layernorm(x, normalized_shape=[3])
    y_grad = ad.Variable("y_grad")
    x_grad = y.op.gradient(y, y_grad)[0]
    evaluator = ad.Evaluator(eval_nodes=[x_grad])

    x_val = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
    y_grad_val = torch.tensor([[12, 4, 2], [-3, -5, 3]], dtype=torch.float32)

    check_evaluator_output(
        evaluator,
        input_values={x: x_val, y_grad: y_grad_val},
        expected_outputs=[
            torch.tensor([
                [1.2248, -2.4495,  1.2246],
                [2.0412, -4.0825, 2.0413]
            ], dtype=torch.float32)
        ]
    )

def test_relu():
    x = ad.Variable("x")
    y = ad.relu(x)
    y_grad = ad.Variable("y_grad")
    x_grad = y.op.gradient(y, y_grad)[0]
    evaluator = ad.Evaluator(eval_nodes=[x_grad])

    x_val = torch.tensor([[-1.0, 2.0, 0.0], [3.0, -4.0, 5.0]], dtype=torch.float32)
    y_grad_val = torch.ones_like(x_val)

    check_evaluator_output(
        evaluator,
        input_values={x: x_val, y_grad: y_grad_val},
        expected_outputs=[torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]], dtype=torch.float32)]
    )

def test_softmax():
    x = ad.Variable("x")
    y = ad.softmax(x)
    y_grad = ad.Variable("y_grad")
    x_grad = y.op.gradient(y, y_grad)[0]
    evaluator = ad.Evaluator(eval_nodes=[x_grad])

    x_val = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
    y_grad_val = torch.tensor([[0.5, -0.3, 0.8], [-0.2, 0.4, -0.1]], dtype=torch.float32)

    check_evaluator_output(
        evaluator,
        input_values={x: x_val, y_grad: y_grad_val},
        expected_outputs=[
            torch.tensor([
                [-0.0003, -0.1967,  0.1971],
                [-0.0192,  0.0946, -0.0754]
            ], dtype=torch.float32)
        ]
    )

def test_transpose():
    x = ad.Variable("x")
    y = ad.transpose(x, 1, 0)
    y_grad = ad.Variable("y_grad")
    x_grad = y.op.gradient(y, y_grad)[0]
    evaluator = ad.Evaluator(eval_nodes=[x_grad])

    x_val = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y_grad_val = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    check_evaluator_output(
        evaluator,
        input_values={x: x_val, y_grad: y_grad_val},
        expected_outputs=[torch.tensor([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]])]
    )

def test_broadcast():
    x = ad.Variable("x")
    y = ad.broadcast(x, input_shape=[3, 2], target_shape=[2, 3, 2])
    y_grad = ad.Variable("y_grad")
    x_grad = y.op.gradient(y, y_grad)[0]
    evaluator = ad.Evaluator(eval_nodes=[x_grad])

    x_val = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y_grad_val = torch.tensor([
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], 
        [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]
    ])

    check_evaluator_output(
        evaluator,
        input_values={x: x_val, y_grad: y_grad_val},
        expected_outputs=[torch.tensor([[8.0, 10.0], [12.0, 14.0], [16.0, 18.0]])]
    )

def test_sqrt():
    x = ad.Variable("x")
    y = ad.sqrt(x)
    y_grad = ad.Variable("y_grad")
    x_grad = y.op.gradient(y, y_grad)[0]
    evaluator = ad.Evaluator(eval_nodes=[x_grad])

    x_val = torch.tensor([[4.0, 9.0], [16.0, 25.0]])
    y_grad_val = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    check_evaluator_output(
        evaluator,
        input_values={x: x_val, y_grad: y_grad_val},
        expected_outputs=[torch.tensor([[0.25, 0.33333], [0.375, 0.4]])]
    )

def test_power():
    x = ad.Variable("x")
    y = ad.power(x, 2)
    y_grad = ad.Variable("y_grad")
    x_grad = y.op.gradient(y, y_grad)[0]
    evaluator = ad.Evaluator(eval_nodes=[x_grad])

    x_val = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y_grad_val = torch.tensor([[1.0, 1.0], [1.0, 1.0]])

    check_evaluator_output(
        evaluator,
        input_values={x: x_val, y_grad: y_grad_val},
        expected_outputs=[torch.tensor([[2.0, 4.0], [6.0, 8.0]])]
    )


def test_mean_keepdim_false():
    x = ad.Variable("x")
    y = ad.mean(x, dim=(1,), keepdim=False)  # 形状 (2,)
    y_grad = ad.Variable("y_grad")
    # dL/dx
    x_grad = y.op.gradient(y, y_grad)[0]
    evaluator = ad.Evaluator(eval_nodes=[x_grad])

    x_val = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                          [-1.0, 1.0, 3.0, 5.0]], dtype=torch.float32)

    # 设外部传入的 dy/dy = y_grad = [2.0, -1.0]
    # keepdim=False 时，梯度需要在被约简维度上均分到每个元素：/4
    y_grad_val = torch.tensor([2.0, -1.0], dtype=torch.float32)

    # 期望 dL/dx:
    # 第一行每个元素 2.0/4 = 0.5
    # 第二行每个元素 -1.0/4 = -0.25
    x_grad_expected = torch.tensor([[0.5, 0.5, 0.5, 0.5],
                                   [-0.25, -0.25, -0.25, -0.25]], dtype=torch.float32)

    check_evaluator_output(
        evaluator,
        input_values={x: x_val, y_grad: y_grad_val},
        expected_outputs=[x_grad_expected],
    )


def test_mean_keepdim_true_multi_dims():
    x = ad.Variable("x")
    # 在 dim=(1,2) 上求平均，keepdim=True -> 输出形状 (2,1,1)
    y = ad.mean(x, dim=(1, 2), keepdim=True)
    y_grad = ad.Variable("y_grad")
    x_grad = y.op.gradient(y, y_grad)[0]
    evaluator = ad.Evaluator(eval_nodes=[x_grad])

    x_val = torch.tensor([[[1.0, 2.0],
                           [3.0, 4.0],
                           [5.0, 6.0]],

                          [[7.0, 8.0],
                           [9.0, 10.0],
                           [11.0, 12.0]]], dtype=torch.float32)  # 形状 (2,3,2)

    # keepdim=True，外部梯度形状与 y 一致：(2,1,1)
    # 令外部梯度为 1，按定义应均分到被约简的 3*2=6 个元素：1/6
    y_grad_val = torch.ones((2, 1, 1), dtype=torch.float32)

    x_grad_expected = torch.full_like(x_val, 1.0 / 6.0)

    check_evaluator_output(
        evaluator,
        input_values={x: x_val, y_grad: y_grad_val},
        expected_outputs=[x_grad_expected],
    )

def test_mean_negative_dim_keepdim_false():
    x = ad.Variable("x")
    # 在最后一维上做 mean，keepdim=False
    y = ad.mean(x, dim=(-1,), keepdim=False)   # (B,)
    y_grad = ad.Variable("y_grad")
    x_grad = y.op.gradient(y, y_grad)[0]
    evaluator = ad.Evaluator(eval_nodes=[x_grad])

    x_val = torch.tensor([[1.,2.,3.,4.],[5.,6.,7.,8.]], dtype=torch.float32)   # (2,4)
    y_grad_val = torch.tensor([3., -2.], dtype=torch.float32)                   # (2,)
    # 每行均分到 4 个元素
    expected = torch.tensor([[0.75,0.75,0.75,0.75], [-0.5,-0.5,-0.5,-0.5]])
    check_evaluator_output(evaluator, {x: x_val, y_grad: y_grad_val}, [expected])


def test_mean_keepdim_true_singleton_axes():
    x = ad.Variable("x")
    # 在 (1,2) 轴上 mean，保留维度 → 输出 (B,1,1)
    y = ad.mean(x, dim=(1,2), keepdim=True)
    y_grad = ad.Variable("y_grad")
    x_grad = y.op.gradient(y, y_grad)[0]
    evaluator = ad.Evaluator(eval_nodes=[x_grad])

    x_val = torch.randn(2, 3, 5)     # (B=2, M=3, N=5)
    y_grad_val = torch.ones(2, 1, 1) # 形状与 y 一致
    N = 3*5
    expected = torch.ones_like(x_val) * (1.0 / N)
    check_evaluator_output(evaluator, {x: x_val, y_grad: y_grad_val}, [expected])


def test_sum_keepdim_false():
    # sum 的梯度应为把外部梯度广播回输入形状（不除以 N）
    x = ad.Variable("x")
    y = ad.sum_op(x, dim=(1,), keepdim=False)    # (B,)
    y_grad = ad.Variable("y_grad")
    x_grad = y.op.gradient(y, y_grad)[0]
    evaluator = ad.Evaluator(eval_nodes=[x_grad])

    x_val = torch.tensor([[1.,2.,3.],[4.,5.,6.]], dtype=torch.float32)  # (2,3)
    y_grad_val = torch.tensor([2., -1.], dtype=torch.float32)           # (2,)
    # 广播到 (2,3)
    expected = torch.tensor([[2.,2.,2.], [-1.,-1.,-1.]], dtype=torch.float32)
    check_evaluator_output(evaluator, {x: x_val, y_grad: y_grad_val}, [expected])


def test_sum_keepdim_true_multi_dims():
    x = ad.Variable("x")
    y = ad.sum_op(x, dim=(1,2), keepdim=True)   # 输出 (B,1,1)
    y_grad = ad.Variable("y_grad")
    x_grad = y.op.gradient(y, y_grad)[0]
    evaluator = ad.Evaluator(eval_nodes=[x_grad])

    x_val = torch.randn(2, 3, 2)
    y_grad_val = torch.tensor([[[3.]], [[-2.]]], dtype=torch.float32)   # (2,1,1)
    expected = y_grad_val.expand_as(x_val)                               # 直接广播
    check_evaluator_output(evaluator, {x: x_val, y_grad: y_grad_val}, [expected])


def test_expand_as_3d_like_mean_case():
    # 仿真 mean keepdim=False 的常见形状：y_grad (B,) 扩到 x (B,T)
    # 这里我们直接测试 expand_as_3d 在反向中常见的广播行为
    a = ad.Variable("a")  # 模拟 y_grad
    b = ad.Variable("b")  # 模拟 x
    node = ad.expand_as_3d(a, b)     # 仅测试 forward 输出是否符合预期
    evaluator = ad.Evaluator(eval_nodes=[node])

    a_val = torch.tensor([2., -1.], dtype=torch.float32)      # (B=2,)
    b_val = torch.zeros(2, 4, dtype=torch.float32)            # (B=2, T=4)
    expected = a_val.unsqueeze(1).expand_as(b_val)            # (2,4)
    # 由于这里没有梯度节点，直接比较前向输出
    out = evaluator.run({a: a_val, b: b_val})[0]
    torch.testing.assert_close(out, expected, atol=1e-4, rtol=1e-4)


def test_transpose_negative_axes():
    x = ad.Variable("x")
    y = ad.transpose(x, -1, -2)
    y_grad = ad.Variable("y_grad")
    x_grad = y.op.gradient(y, y_grad)[0]
    evaluator = ad.Evaluator(eval_nodes=[x_grad])

    x_val = torch.randn(2, 3, 4)
    y_grad_val = torch.randn(2, 4, 3)
    expected = y_grad_val.transpose(-1, -2)  # 反向再转回
    check_evaluator_output(evaluator, {x: x_val, y_grad: y_grad_val}, [expected])


if __name__ == "__main__":
    test_mul()
    test_div()
    test_div_by_const()
    test_layernorm()
    test_relu() 
    test_softmax()
    test_matmul()
    test_matmul_3d()
    test_transpose()
    test_broadcast()
    test_sqrt()
    test_power()
    test_mean_keepdim_false()
    test_mean_keepdim_true_multi_dims()
    test_mean_negative_dim_keepdim_false()
    test_mean_keepdim_true_singleton_axes()
    test_sum_keepdim_false()
    test_sum_keepdim_true_multi_dims()
    test_expand_as_3d_like_mean_case()
    test_transpose_negative_axes()
