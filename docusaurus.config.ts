import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  title: 'AI Book',
  tagline: '从基础到实践，系统掌握人工智能',
  favicon: 'img/favicon.ico',

  future: {
    v4: true,
  },

  url: 'https://ai-book.example.com',
  baseUrl: '/',

  markdown: {
    mermaid: true,
  },

  onBrokenLinks: 'throw',

  i18n: {
    defaultLocale: 'zh-Hans',
    locales: ['zh-Hans', 'en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
        },
        blog: {
          showReadingTime: true,
          feedOptions: {
            type: ['rss', 'atom'],
            xslt: true,
          },
          blogTitle: 'AI 动态',
          blogDescription: '人工智能领域最新动态与思考',
          onInlineTags: 'warn',
          onInlineAuthors: 'warn',
          onUntruncatedBlogPosts: 'warn',
        },
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themes: ['@docusaurus/theme-mermaid'],

  themeConfig: {
    image: 'img/ai-book-social-card.jpg',
    colorMode: {
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: 'AI Book',
      logo: {
        alt: 'AI Book Logo',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'mlBasicsSidebar',
          label: '机器学习基础',
          position: 'left',
        },
        {
          type: 'docSidebar',
          sidebarId: 'llmSidebar',
          label: '大语言模型',
          position: 'left',
        },
        {
          type: 'docSidebar',
          sidebarId: 'promptSidebar',
          label: '提示工程',
          position: 'left',
        },
        {
          type: 'docSidebar',
          sidebarId: 'aiAppSidebar',
          label: 'AI 应用开发',
          position: 'left',
        },
        {to: '/blog', label: 'AI 动态', position: 'left'},
        {
          type: 'localeDropdown',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: '知识库',
          items: [
            {
              label: '机器学习基础',
              to: '/docs/ml-basics/overview',            },
            {
              label: '大语言模型',
              to: '/docs/llm/overview',
            },
            {
              label: '提示工程',
              to: '/docs/prompt-engineering/overview',
            },
            {
              label: 'AI 应用开发',
              to: '/docs/ai-application/overview',
            },
          ],
        },
        {
          title: '社区',
          items: [
            {
              label: 'AI 动态',
              to: '/blog',
            },
          ],
        },
        {
          title: '更多',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/your-org/ai-book',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} AI Book. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['python', 'bash', 'json'],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
