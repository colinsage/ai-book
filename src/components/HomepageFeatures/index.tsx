import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  description: ReactNode;
  to: string;
  icon: string;
};

const FeatureList: FeatureItem[] = [
  {
    title: '机器学习基础',
    icon: '🧠',
    to: '/docs/ml-basics/overview',
    description: (
      <>
        系统学习监督学习、无监督学习、模型评估与特征工程等核心概念，构建扎实的
        AI 基础。
      </>
    ),
  },
  {
    title: '大语言模型',
    icon: '🤖',
    to: '/docs/llm/overview',
    description: (
      <>
        深入理解 Transformer 架构、预训练与微调、推理优化，以及
        GPT、LLaMA 等主流模型的原理与应用。
      </>
    ),
  },
  {
    title: '提示工程',
    icon: '✨',
    to: '/docs/prompt-engineering/overview',
    description: (
      <>
        掌握 Chain-of-Thought、Few-shot
        Learning、Agent 提示模式等高级技巧，充分发挥大模型能力。
      </>
    ),
  },
  {
    title: 'AI 应用开发',
    icon: '🚀',
    to: '/docs/ai-application/overview',
    description: (
      <>
        实战 RAG 检索增强生成、LangChain 框架、AI Agent
        开发，以及主流工具与生态的快速上手。
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
