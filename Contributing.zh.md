# 贡献

感谢您考虑为 `torch-supa` 项目贡献力量。本指南将解释如何报告问题、提交更改以及如何帮助维护一致的贡献流程。

## 行为准则

所有贡献者都应遵守[贡献者公约](https://www.contributor-covenant.org/version/3/0/code_of_conduct/)。任何不当行为、骚扰或人身攻击都将不被容忍。

## 报告问题

在提交问题之前，请先检查是否已存在类似问题。

创建新问题时，请尽可能提供所有相关信息：

- 问题或所需改进的清晰描述
- 重现问题的步骤，最好提供最小可复现示例
- 预期行为和实际行为
- 环境详情，例如操作系统、Python 版本、PyTorch 版本、torch-supa 版本、提交 ID、驱动程序/运行时版本和硬件型号
- 相关代码片段、日志、堆栈跟踪或屏幕截图

对于安全漏洞，请遵循 [Security Policy.md](Security%20Policy.md) 中的报告流程。

## 提交代码

1. **fork** 该仓库到您自己的帐户。
2. **从目标分支创建特性分支**：

```bash

git checkout -b feature/your-feature-name

```

3. **按照周围代码的风格和约定进行更改**。
4. **添加或更新测试**，以支持新功能、错误修复和行为变更（如可行）。
5. **提交前运行相关测试**。例如​​：

```bash

cd torch-supa/test && python3 start_test.py test_abs_kernel.py

```

6. **提交更改**：

```bash

git commit -m "fix: resolve issue #123"

```

7. **推送**你的分支：

```bash

git push origin feature/your-feature-name

```

8. **针对此仓库的目标分支发起拉取请求 (Pull Request)**。

## 贡献者许可协议（CLA）

本项目要求所有贡献者在合并其贡献之前，必须签署贡献者许可协议（Contributor License Agreement，简称“CLA”）。签署 CLA 即表示您授予本项目在项目开源许可下使用、修改和分发您所贡献内容的权利，同时您确认您是贡献内容的原创作者，或拥有提交该内容的合法权利。

如果您是以个人身份独立参与本项目贡献（不涉及任何雇主或第三方机构的职务成果），请签署个人贡献者许可协议；若您是代表您所在的雇主、公司、政府机构或其他法律实体进行贡献，或所提交的贡献内容属于您的职务开发成果，则必须签署公司贡献者许可协议。一旦您的 CLA 签署记录存档，您即可正常提交 Pull Request，无需额外操作。如果您的 CLA 尚未签署，项目维护者可能会在审核您的贡献之前要求您先完成签署。

在提交 Pull Request 时，请在 Pull Request 描述中包含一条声明，确认您已阅读并同意 CLA 的条款（例如：“I have signed the CLA and agree to its terms.”）。项目也可能使用自动化机器人来验证 CLA 的签署状态。

如果您对 CLA 有任何疑问，请联系 [填写项目联系邮箱]。

## 测试要求

添加新功能或修复错误时，请尽可能添加相应的测试。至少，运行与您更改的文件或功能相关的测试。如果无法添加测试或无法在本地运行测试，请在拉取请求描述中说明原因。

## Pull Request 指南

一个好的 Pull Request 应包含：

- 变更的简明摘要
- 变更动机或待解决的问题
- 已执行的测试及其结果
- 任何已知的限制、风险或后续工作
- 相关问题、设计说明或讨论的链接（如有）

对于重大变更，请先创建一个 issue 来讨论拟议的设计及其预期影响。

## Pull Request 审核

- Pull Request 需要维护者审核才能合并。
- 审核者可能会要求进行正确性、可维护性、兼容性、文档、测试或安全性方面的修改。
- 长时间无活动的 Pull Request 可能会被关闭。当工作恢复时，可以重新打开。

## 许可

通过贡献代码，您同意您的贡献将根据本仓库 LICENSE 文件中指定的开源许可进行许可。如果您对许可有任何疑问，请联系维护者。

## 联系方式

如有任何疑问，请发送邮件至 [opensource@birentech.com](mailto:opensource@birentech.com) 或使用代码仓库的常规问题反馈或讨论流程。

感谢您的贡献！

[贡献者指南-中文版](https://my.feishu.cn/wiki/C7tUwRlUAiFgtTk42HicgcjUnVe)
