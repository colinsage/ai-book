import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import Heading from '@theme/Heading';
import {translate} from '@docusaurus/Translate';
import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  description: ReactNode;
  to: string;
  icon: string;
};

const FeatureList: FeatureItem[] = [
  {
    title: translate({message: '模型原理'}),
    icon: '🧠',
    to: '/docs/llm/architecture',
    description: (
      <>
        {translate({message: '深入理解 Transformer 架构、自注意力机制、位置编码，以及仅解码器架构的设计思想。'})}
      </>
    ),
  },
  {
    title: translate({message: '训练与推理'}),
    icon: '🤖',
    to: '/docs/llm/training',
    description: (
      <>
        {translate({message: '掌握预训练与微调流程、LoRA/QLoRA 高效微调、RLHF 对齐，以及量化与推理加速技术。'})}
      </>
    ),
  },
  {
    title: translate({message: '提示工程'}),
    icon: '✨',
    to: '/docs/llm/prompt-engineering',
    description: (
      <>
        {translate({message: '掌握 Zero-shot、Few-shot、Chain-of-Thought 等核心技巧，充分发挥大模型能力。'})}
      </>
    ),
  },
  {
    title: translate({message: '应用开发'}),
    icon: '🚀',
    to: '/docs/llm/rag',
    description: (
      <>
        {translate({message: '实战 RAG 检索增强生成、AI Agent 开发、LangChain 框架，快速构建 AI 应用。'})}
      </>
    ),
  },
];

function Feature({title, icon, description, to}: FeatureItem) {
  return (
    <div className={clsx('col col--3')}>
      <Link to={to} className={styles.featureCard}>
        <div className="text--center">
          <span className={styles.featureIcon}>{icon}</span>
        </div>
        <div className="text--center padding-horiz--md">
          <Heading as="h3">{title}</Heading>
          <p>{description}</p>
        </div>
      </Link>
    </div>
  );
}

export default function HomepageFeatures(): ReactNode {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
