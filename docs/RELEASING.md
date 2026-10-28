# 发布到 PyPI

推送版本标签后，GitHub Actions 自动运行 Python 3.11 / 3.12 / 3.13 CI、严格文档构建、构建 wheel/sdist 和 Twine 校验，然后通过 OIDC 发布。没有 PyPI Token，也不需要新增 GitHub Secret。

## 首次配置（项目所有者）

1. 在 GitHub 仓库 Settings → Environments 创建 `pypi` 环境。建议限制只能由 `v*` 标签部署，并设置 required reviewers（账号方案支持时）。保护可创建发布标签的权限。
2. 打开 https://pypi.org/manage/project/wuwei/settings/publishing/ ，添加 GitHub Trusted Publisher，逐项填写：
   - Owner：`xiaojiaenen`
   - Repository：`wuwei`
   - Workflow filename：`release.yml`（不是路径，也不是 CI 工作流名称）
   - Environment：`pypi`
3. 如果项目尚未创建，改在 https://pypi.org/manage/account/publishing/ 注册 pending publisher，项目名填 `wuwei`；名称必须可用。

官方说明：[配置身份](https://docs.pypi.org/trusted-publishers/adding-a-publisher/)、[OIDC 发布](https://docs.pypi.org/trusted-publishers/using-a-publisher/)。

## 验证配置文件，不上传

先将工作流合并到默认分支 `master`，再在 GitHub Actions → Publish to PyPI → Run workflow 选择待验证分支。手动运行只测试、构建和存储产物，**不发布**；GitHub 要求工作流先存在于默认分支，才能手动触发。

## 正式发布

先将发布代码与工作流合并到 `master`，确认 `pyproject.toml` 版本、更新日志及 CI：

```bash
git switch master
git pull --ff-only origin master
# 当前项目版本为 3.2.1；后续按实际版本修改标签。
git tag -a v3.2.1 -m 'Release 3.2.1'
git push origin v3.2.1
```

在 Actions 查看 Publish to PyPI，必要时批准 `pypi` 环境部署。推送标签会触发正式上传；不要在绑定完成前推送。标签必须严格等于 `v` + `project.version`，否则发布会被阻止。

上传后在 https://pypi.org/project/wuwei/ 确认版本，并在独立环境安装验证：

```bash
uv run --isolated --no-project --with 'wuwei==3.2.1' \
  python -c "from importlib.metadata import version; print(version('wuwei'))"
```

PyPI 不允许覆盖同名版本文件。失败时先检查 PyPI 是否已收到部分或全部文件，不要删除标签后盲目重发；已存在的版本通常应修正后递增版本发布。工作流不启用 skip-existing，避免隐藏发布冲突。
