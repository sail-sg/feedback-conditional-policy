# Feedback Conditional Policy (FCP) 修改文档

本文档记录了为实现 Feedback Conditional Policy，基于 verl 框架 v0.6.1 版本所做的主要修改。

## 修改概述

为了支持 FCP 训练算法，我们对 verl 框架进行了以下几个方面的修改和扩展：

## 1. recipe/fcp 文件夹

**路径**: `recipe/fcp/`

**修改内容**:
- 实现了 FCP 的配置管理（config）
- 定义了 FCP 的完整训练算法逻辑

**说明**: 该文件夹包含了 FCP 算法的核心实现，包括训练超参数配置、训练流程编排等。

## 2. verl/trainer/ppo/core_algos.py

**路径**: `verl/trainer/ppo/core_algos.py`

**修改内容**:
- 增加了 NLL (Negative Log-Likelihood) loss
- 新增了适用于 FCP 的 advantage estimator

**说明**: 
- FCP 的 advantage estimator 实际上不计算任何 advantage 值
- 此实现主要是为了适配 verl 框架的接口要求
- 通过这种方式，FCP 可以无缝集成到现有的 verl PPO 训练流程中

## 3. verl/utils/reward_score/math_verify.py

**路径**: `verl/utils/reward_score/math_verify.py`

**修改内容**:
- 修改答案验证逻辑，只取生成文本的最后 300 个字符

**说明**: 
- 此修改与评测标准对齐
- 将截取后的文本交给 math-verify 库进行数学答案正确性判断

## 4. verl/workers/reward_manager/gpt_critique_math_score.py

**路径**: `verl/workers/reward_manager/gpt_critique_math_score.py`

**修改内容**:
- 实现了调用外部 reward model 的功能
- 具体使用 GPT-5-nano 模型进行 critique 评分

**说明**: 
- 该模块作为 reward manager，负责获取外部模型的反馈
- GPT-5-nano 用于生成对模型输出的批评性反馈（critique）
- 这些反馈作为训练信号指导 FCP 的策略优化

## 5. verl/trainer/main_ppo.py

**路径**: `verl/trainer/main_ppo.py`

**修改内容**:
- 修改 `val_reward_func` 配置
- 将验证时的 reward function 设置为指定的 `NaiveRewardManager`

**说明**: 
- 确保在验证阶段使用正确的 reward 计算方式
- `NaiveRewardManager` 提供了适合 FCP 验证的 reward 计算逻辑

## 兼容性说明

- 基础框架版本: verl v0.6.1
- 所有修改均保持与原有 verl 接口的兼容性
- 可以通过配置文件轻松切换 FCP 和其他训练算法

## 使用方法

请参考 `recipe/fcp/` 目录下的训练脚本和配置文件，了解如何启动 FCP 训练任务。

---

**最后更新**: 2026-01-05

