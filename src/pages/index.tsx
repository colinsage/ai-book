import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import Heading from '@theme/Heading';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <Heading as="h1" className="hero__title">
          {siteConfig.title}
        </Heading>
        <p className={styles.heroSubtitle}>{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/docs/llm/overview">
            大语言模型
          </Link>
          <Link
            className="button button--secondary button--lg"
            to="/docs/llama-cpp/index">
            Llama.cpp 实现原理
          </Link>
          <Link
            className="button button--secondary button--lg"
            to="/docs/sglang/index">
            SGLang 实现原理
          </Link>
          <Link
            className="button button--secondary button--lg"
            to="/docs/vllm/index">
            vLLM 实现原理
          </Link>
        </div>
      </div>
    </header>
  );
}

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`欢迎来到 ${siteConfig.title}`}
      description="从基础到实践，系统掌握人工智能 — 涵盖大语言模型原理、提示工程、AI应用开发与LLM推理引擎源码分析">
      <HomepageHeader />
      <main>
        <HomepageFeatures />
      </main>
    </Layout>
  );
}
