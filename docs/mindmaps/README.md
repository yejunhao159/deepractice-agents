# PlantUML 图表库

本目录包含《多智能体系统开发实战》教材的完整 PlantUML 图表，每个章节使用多种图表类型（思维导图、时序图、类图、状态图、活动图等）来更好地表达内容。

## 目录结构

```
mindmaps/
├── overview/           # 总览图
│   └── 01-full-curriculum.puml    # 完整课程体系
│
├── chapter01/          # 第1章：初识智能体
│   ├── 01-overview.puml           # 知识全景图
│   ├── 02-agent-loop.puml         # Agent Loop 时序图
│   ├── 03-agent-architecture.puml # 四层架构组件图
│   ├── 04-agent-types.puml        # 智能体类型状态图
│   └── 05-ecosystem.puml          # 应用生态全景图
│
├── chapter02/          # 第2章：智能体发展史
│   ├── 01-overview.puml           # 知识全景图
│   ├── 02-evolution-timeline.puml # 演进时间线
│   ├── 03-minsky-society.puml     # 心智社会理论
│   └── 04-paradigm-shift.puml     # 范式演进对比
│
├── chapter03/          # 第3章：大语言模型与智能体
│   ├── 01-overview.puml           # 知识全景图
│   ├── 02-transformer.puml        # Transformer 架构
│   ├── 03-model-selection.puml    # 模型选型决策树
│   └── 04-rag-vs-memory.puml      # RAG vs 记忆对比
│
├── chapter04/          # 第4章：PromptX 平台
│   ├── 01-overview.puml           # 知识全景图
│   ├── 02-semantic-gap.puml       # 语义鸿沟识别活动图
│   ├── 03-nuwa-role.puml          # Nuwa 角色创建时序图
│   ├── 04-engram-network.puml     # Engram 语义网络
│   └── 05-cognitive-cycle.puml    # 认知循环状态图
│
├── chapter05/          # 第5章：AgentX 框架
│   ├── 01-overview.puml           # 知识全景图
│   ├── 02-mealy-machine.puml      # Mealy Machine 状态图
│   ├── 03-event-layers.puml       # 四层事件模型
│   ├── 04-architecture.puml       # 系统架构组件图
│   └── 05-integration-flow.puml   # PromptX 集成时序图
│
├── chapter06/          # 第6章：Deepractice 方法论
│   ├── 01-overview.puml           # 知识全景图
│   ├── 02-4p-theory.puml          # 4P 理论转化链
│   ├── 03-task-state-machine.puml # AI 任务状态机
│   ├── 04-pateoas.puml            # PATEOAS 状态驱动
│   └── 05-ai-organization.puml    # AI 组织化演进
│
├── chapter07/          # 第7章：智能体经典范式
│   ├── 01-overview.puml           # 知识全景图
│   ├── 02-react-flow.puml         # ReAct 执行流程
│   └── 03-paradigm-comparison.puml # 三种范式对比
│
├── chapter08/          # 第8章：主流框架对比
│   └── 01-overview.puml           # 知识全景图
│
├── chapter09/          # 第9章：Monogent 认知架构
│   ├── 01-overview.puml           # 知识全景图
│   └── 02-seven-stages.puml       # 七阶段认知管道
│
├── chapter10/          # 第10章：上下文工程
│   └── 01-overview.puml
│
├── chapter11/          # 第11章：工具使用
│   └── 01-overview.puml
│
├── chapter12/          # 第12章：多智能体协作
│   └── 01-overview.puml
│
├── chapter13/          # 第13章：代码助手实战
│   └── 01-overview.puml
│
├── chapter14/          # 第14章：知识问答系统
│   └── 01-overview.puml
│
├── chapter15/          # 第15章：自动化工作流
│   └── 01-overview.puml
│
└── chapter16/          # 第16章：总结与展望
    └── 01-overview.puml
```

## 图表类型说明

| 类型 | 语法 | 用途 |
|-----|------|------|
| **Mind Map** | `@startmindmap` | 知识全景图、概念关系 |
| **Sequence** | `@startuml` + participant | 流程交互、时序关系 |
| **Class/Component** | `@startuml` + component | 系统架构、组件关系 |
| **State** | `@startuml` + state | 状态机、生命周期 |
| **Activity** | `@startuml` + activity | 工作流程、决策树 |
| **Timeline** | `@startuml` + concise | 时间线、演进历史 |

## 使用方法

### 在线渲染
访问 [PlantUML Server](https://www.plantuml.com/plantuml/uml)，粘贴代码即可预览。

### VS Code 插件
```bash
# 安装插件后，打开 .puml 文件按 Alt+D 预览
code --install-extension jebbs.plantuml
```

### 命令行生成
```bash
# 安装 (macOS)
brew install plantuml

# 生成单个文件
plantuml chapter01/01-overview.puml

# 批量生成 PNG
find . -name "*.puml" -exec plantuml {} \;

# 生成 SVG
plantuml -tsvg chapter01/*.puml
```

### Docker
```bash
docker run -v $(pwd):/data plantuml/plantuml chapter01/01-overview.puml
```

## 样式说明

### Mind Map 样式
```plantuml
<style>
mindmapDiagram {
  :depth(0) { BackgroundColor #3498DB; FontColor white; FontSize 18 }
  :depth(1) { BackgroundColor #2ECC71; FontColor white; FontSize 14 }
  :depth(2) { BackgroundColor #F39C12; FontColor white; FontSize 12 }
}
</style>
```

### 常用颜色
| 颜色代码 | 用途 |
|---------|------|
| `#E74C3C` | 重点/警告 |
| `#3498DB` | 主要/默认 |
| `#2ECC71` | 成功/完成 |
| `#F39C12` | 注意/进行中 |
| `#9B59B6` | 特殊/高级 |
| `#2C3E50` | 标题/深色 |

## 扩展建议

每个章节可以根据需要添加更多图表：
- **时序图**：展示交互流程
- **类图**：展示代码结构
- **ER图**：展示数据模型
- **甘特图**：展示项目计划

## 参考资源

- [PlantUML 官方文档](https://plantuml.com/zh/)
- [PlantUML Mind Map](https://plantuml.com/zh/mindmap-diagram)
- [PlantUML 实时预览](https://www.plantuml.com/plantuml/uml)
