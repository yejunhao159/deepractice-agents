# 飞书文档同步配置指南

本指南帮助你配置 CI/CD 管道，自动将 Markdown 文档同步到飞书知识库。

## 1. 创建飞书应用

1. 打开 [飞书开放平台](https://open.feishu.cn/app)
2. 点击「创建企业自建应用」
3. 填写应用名称（如：文档同步机器人）
4. 创建后进入应用详情页

## 2. 配置应用权限

在应用详情页 → 「权限管理」中添加以下权限：

| 权限名称 | 权限标识 |
|---------|---------|
| 查看、创建、编辑和管理知识库 | `wiki:wiki` |
| 查看、评论、编辑和管理文档 | `docx:document` |

添加完成后点击「申请开通」，等待管理员审批。

## 3. 获取凭证

### App ID 和 App Secret
在应用详情页 → 「凭证与基础信息」中获取：
- **App ID**: 应用唯一标识
- **App Secret**: 应用密钥（请妥善保管）

### 知识库 Space ID
1. 打开飞书知识库
2. 进入目标知识库
3. 从 URL 中获取 Space ID：
   ```
   https://xxx.feishu.cn/wiki/space/xxxxx ← 这个 xxxxx 就是 Space ID
   ```

## 4. 配置 GitHub Secrets

在你的 GitHub 仓库中：

1. 进入 `Settings` → `Secrets and variables` → `Actions`
2. 点击 `New repository secret`
3. 添加以下 Secrets：

| Secret 名称 | 值 |
|------------|-----|
| `FEISHU_APP_ID` | 你的飞书 App ID |
| `FEISHU_APP_SECRET` | 你的飞书 App Secret |
| `FEISHU_SPACE_ID` | 目标知识库的 Space ID |

## 5. 触发同步

### 自动触发
当 `docs/` 目录有变更推送到 `main` 分支时，会自动触发同步。

### 手动触发
1. 进入 GitHub 仓库的 `Actions` 页面
2. 选择「同步文档到飞书知识库」工作流
3. 点击 `Run workflow`
4. 可选择自定义要同步的目录

## 6. 本地测试

```bash
# 设置环境变量
export FEISHU_APP_ID="your_app_id"
export FEISHU_APP_SECRET="your_app_secret"
export FEISHU_SPACE_ID="your_space_id"
export DOCS_DIR="docs"

# 运行同步脚本
node scripts/sync-to-feishu.js
```

## 常见问题

### Q: 提示 token 获取失败？
A: 检查 App ID 和 App Secret 是否正确，应用是否已发布。

### Q: 提示没有权限？
A: 确认应用权限已申请并通过审批，知识库管理员已授权该应用。

### Q: 文档内容格式不正确？
A: 当前脚本使用简化的 Markdown 转换，复杂格式（表格、图片等）可能需要额外处理。

## 高级配置

### 自定义同步目录
修改 `.github/workflows/sync-to-feishu.yml` 中的 `paths`：

```yaml
on:
  push:
    paths:
      - 'docs/**'
      - 'blog/**'  # 添加更多目录
```

### 定时同步
添加定时触发：

```yaml
on:
  schedule:
    - cron: '0 2 * * *'  # 每天凌晨 2 点
```
