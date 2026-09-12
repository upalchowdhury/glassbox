/* Course navigation and plain-language goals are content, not rendering logic. */
(function attachCourseContent(root) {
  root.GlassboxCourseContent = {
    modules: [
      ['tinygpt', 'How models learn', 'Pretraining from scratch'],
      ['moe', 'Mixture of experts', 'Route tokens through experts'],
      ['posttrain', 'Teaching and adapting', 'SFT, LoRA, and preferences'],
      ['environment', 'Define success', 'Design the task before RL'],
      ['grpo', 'Learning from rewards', 'GRPO and verifiable rewards'],
      ['agents', 'From answers to agents', 'Inspect full tool trajectories'],
      ['scaling', 'Scaling the system', 'Workers, queues, and recovery'],
      ['dashboard', 'Put it all together', 'Diagnosis and conceptual review'],
    ],
    depth: {
      see: 'Summary', explain: 'Read', derive: 'Deep dive',
    },
  };
}(typeof window !== 'undefined' ? window : globalThis));
