# Security Policy

## Supported Versions

We support the latest maintained releases listed below. Older releases are not back-patched unless explicitly stated by the maintainers.

| Version | Supported |
| --- | --- |
| 2.12.0 | √ |
| 2.11.0 | √ |
| 2.10.0 | √ |
| 2.9.0 | √ |
| 2.8.0 | √ |
| 2.6.0 | √ |
| < 2.6.0 | × |

## Reporting a Vulnerability

**Security vulnerabilities may be reported through public issues, public pull requests, public discussions, or the private reporting channels listed below.**

Use the reporting channel that best fits the sensitivity of the issue:

- **Public issues, public pull requests, or public discussions**: suitable for non-sensitive security reports, hardening suggestions, documentation issues, or issues that do not expose exploitable details.
- **Github Private Vulnerability Reporting**: `[Insert your Gitee /security/advisories URL]`
- **Private email**: [opensource@birentech.com](mailto:opensource@birentech.com) (if private vulnerability reporting is not available)

When reporting, please include as much of the following information as possible:

- A clear description of the vulnerability and its potential impact
- Steps to reproduce, preferably with a minimal reproduction
- Affected version(s), such as PyTorch version, torch-supa commit ID
- Environment details, such as OS, Python version, LKG version, driver/runtime version, and hardware model
- Whether the issue is exploitable in a default configuration
- Suggested mitigation or patch, if available

Please avoid including sensitive credentials, production secrets, private customer data, or unrelated personal information in your report.

## Security Note on PyTorch Models

This project depends on PyTorch. Please be aware that **PyTorch models are programs**, and running untrusted models is equivalent to running untrusted code. If you use this project for model loading or inference, only use model files (for example, `.pth` or `.pt`) from trusted sources. Loading models from untrusted sources may lead to arbitrary code execution, which is a general risk of PyTorch and projects built on top of it.

## What to Expect

After receiving a vulnerability report, we aim to follow this process:

1. **Acknowledgment**: we will acknowledge receipt within 5 business days on a best-effort basis.
2. **Triage**: we will confirm whether the report is in scope and may request additional information.
3. **Assessment**: we will evaluate severity, affected versions, exploitability, and possible mitigations.
4. **Remediation**: if the issue is confirmed, we will work on a fix or mitigation plan.
5. **Coordinated disclosure**: we will coordinate with the reporter on an appropriate disclosure timeline before publishing an advisory.

> **Legal note**: The above timelines and process are targets only. The project maintainers make no contractual or legal guarantee of response, remediation, or disclosure timelines. Nothing in this policy creates an obligation enforceable against the project or its contributors.

## Scope

**In scope** vulnerabilities include security issues in the following repository components:

- `torch_supa/`
- `_inductor/`
- `backends/`
- `cmake/`
- `contrib/`
- `csrc/`
- `distributed/`
- `multiprocessing/`
- `profiler/`
- `supa/`
- `testing/`
- `utils/`
- `docs/`

Examples of in-scope issues may include:

- Memory corruption or unsafe memory access in project-maintained native code
- Privilege boundary violations introduced by this project
- Unsafe handling of inputs that can lead to code execution in a supported configuration
- Security-sensitive misconfiguration in project-provided build, runtime, or packaging logic

**Out of scope** issues include:

- Vulnerabilities in third-party dependencies that we cannot directly fix. Please report those issues upstream. If a viable mitigation exists at this project layer, we may still track the issue here.
- Vulnerabilities in PyTorch itself, which should be reported directly to the PyTorch community.
- Security issues caused by loading untrusted or maliciously crafted model files. See [Security Note on PyTorch Models](#security-note-on-pytorch-models).
- Issues requiring physical access to a developer machine or a compromised local environment.
- Theoretical attacks without a practical exploit path against a default deployment of this software.
- Misconfigurations or vulnerabilities in self-hosted environments that deviate from documented secure defaults.
- Social engineering or phishing targeting project maintainers or users.
- Denial-of-service reports based only on excessive resource consumption without a realistic security impact.

## Responsible Research Guidelines

When testing or reporting security issues, please:

- Test only against systems and environments you are authorized to use.
- Avoid privacy violations, data destruction, service disruption, and degradation of production services.
- Do not exfiltrate data beyond what is necessary to demonstrate the issue.
- Give maintainers a reasonable opportunity to investigate and remediate before public disclosure.

## Recommended Hardening for Forks and Self-Hosted Deployments

If you fork or self-host this project, we recommend enabling the following security features in your repository and deployment environment:

- Private vulnerability reporting
- Code quality and security scanning
- Dependency security scanning
- Branch protection rules
- Required review for pull requests
- CI checks before merge
- Restricted access to release and deployment credentials
- Regular dependency and base image updates

These recommendations do not create a duty of care on our part. You are solely responsible for securing your own deployment.

## Contact

For security reports, use the private reporting channels above. For non-security questions, use the repository's normal issue or discussion process.

[安全政策-中文版](https://my.feishu.cn/wiki/PHpFwpWAPi22LLk5fY5cG873nQf)
