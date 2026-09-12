/* Course navigation and plain-language goals are content, not rendering logic. */
(function attachCourseContent(root) {
  root.GlassboxCourseContent = {
    modules: [
      ['tinygpt', 'TinyGPT', 'Pretraining from scratch'],
      ['moe', 'MoE', 'Route tokens through experts'],
      ['posttrain', 'SFT + LoRA', 'Teach and adapt the base model'],
      ['environment', 'Environment', 'Design the task before RL'],
      ['grpo', 'GRPO / RLVR', 'Learn only from a valid environment'],
      ['agents', 'Agent harness', 'Evaluate full tool trajectories'],
      ['scaling', 'Scaling', 'Balance workers, queues, and recovery'],
      ['dashboard', 'Journey dashboard', 'Compare every checkpoint'],
    ],
    depth: {
      see: 'See it', explain: 'Explain it', derive: 'Derive it',
    },
  };
}(typeof window !== 'undefined' ? window : globalThis));
