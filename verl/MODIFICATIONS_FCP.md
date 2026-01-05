# Feedback Conditional Policy (FCP) Modifications Documentation

**中文版**: [MODIFICATIONS_FCP_zh.md](./MODIFICATIONS_FCP_zh.md)

This document records the major modifications made to the verl framework v0.6.1 to implement Feedback Conditional Policy.

## Modification Overview

To support the FCP training algorithm, we made the following modifications and extensions to the verl framework:

## 1. recipe/fcp Folder

**Path**: `recipe/fcp/`

**Modifications**:
- Implemented FCP configuration management (config)
- Defined the complete FCP training algorithm logic

**Description**: This folder contains the core implementation of the FCP algorithm, including training hyperparameter configuration, training workflow orchestration, etc.

## 2. verl/trainer/ppo/core_algos.py

**Path**: `verl/trainer/ppo/core_algos.py`

**Modifications**:
- Added NLL (Negative Log-Likelihood) loss
- Added a new advantage estimator suitable for FCP

**Description**: 
- The FCP advantage estimator does not actually compute any advantage values
- This implementation is mainly to adapt to the verl framework's interface requirements
- Through this approach, FCP can be seamlessly integrated into the existing verl PPO training pipeline

## 3. verl/utils/reward_score/math_verify.py

**Path**: `verl/utils/reward_score/math_verify.py`

**Modifications**:
- Modified answer verification logic to only take the last 300 characters of the generated text

**Description**: 
- This modification aligns with our evaluation code
- The truncated text is passed to the math-verify library for mathematical answer correctness judgment

## 4. verl/workers/reward_manager/gpt_critique_math_score.py

**Path**: `verl/workers/reward_manager/gpt_critique_math_score.py`

**Modifications**:
- Implemented functionality to call external reward models
- Specifically uses GPT-5-nano model for critique scoring

**Description**: 
- This module serves as a reward manager, responsible for obtaining feedback from external models
- GPT-5-nano is used to generate critical feedback (critique) on model outputs
- These feedbacks serve as training signals to guide FCP policy optimization

## 5. verl/trainer/main_ppo.py

**Path**: `verl/trainer/main_ppo.py`

**Modifications**:
- Modified `val_reward_func` configuration
- Set the reward function during validation to the specified `NaiveRewardManager`

**Description**: 
- Ensures the correct reward calculation method is used during the validation phase
- `NaiveRewardManager` provides reward calculation logic suitable for FCP validation

## Compatibility Notes

- Base framework version: verl v0.6.1
- All modifications maintain compatibility with the original verl interfaces

## Usage

Please refer to the training scripts and configuration files in the `recipe/fcp/` directory to learn how to launch FCP training tasks.

---

**Last Updated**: 2026-01-05