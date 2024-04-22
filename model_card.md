# Model Card: Friends TV Show Script Generator

## 1 Model Details
- Developer: This model was developed by Rachel Montgomery and Sophia Tannir
- Model Version: The model is based on the GPT-2 architecture and was fine-tuned on a dataset of Friends TV show scripts
	- **Fine-tuned from:** GPT-2

## 2 Intended Use
- In Scope:
  - Generating Friends-like TV show scripts based on user-provided scene prompts
  - Entertaining users by creating humorous and engaging scripts reminiscent of the Friends TV show
- Intended Users:
  - Friends TV show fans who enjoy reading fanfiction or alternative storylines
  - Machine learning fans who want to explore the limits of creative writing generation
  - Writers or content creators looking for inspiration or fun writing prompts
- Out of Scope:
  - Using the generated scripts for commercial purposes without proper licensing or permission
  - Claiming the generated scripts as original content created by the user

## 3 Factors
- The quality and coherence of the generated scripts may vary based on the provided scene prompt and its similarity to the training data
- The model's performance may be influenced by the length of the generated script (controlled by the `gen_length` parameter)
- The randomness of the generated scripts can be adjusted using the `temperature`, `top_k`, and `top_p` parameters during generation

## 4 Metrics
- Human Evaluation: Qualitative assessment of the generated scripts by human readers for coherence, engagement, and resemblance to the Friends TV show style

## 5 Training Data
- The model was trained on a dataset of Friends TV show scripts, which were preprocessed and tokenized before training

## 6 Ethical Considerations
- The model was trained on the Friends TV show scripts, which are protected by copyright licensing and permissions and is for educational and exploration purposes only
- The generated scripts should not be used to impersonate or mislead others into believing they are original Friends TV show content
- Users should be aware that the generated scripts are fictional and may contain humorous or exaggerated content not suitable for all ages and audiences

## 7 Caveats and Recommendations
- The model's performance may degrade when given prompts that are significantly different from the Friends TV show style or context
- We recommended that you use the model for entertainment and inspiration purposes only and not rely on it for critical or sensitive content generation
- The generated scripts may occasionally contain repetitive or inconsistent content, which can be helped by adjusting the generation parameters or post-processing the output
