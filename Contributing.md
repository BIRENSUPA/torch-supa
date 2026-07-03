# Contributing

Thank you for considering contributing to `torch-supa`. This guide explains how to report issues, submit changes, and help maintain a consistent contribution process.

## Code of Conduct

All contributors are expected to follow the [Contributor Covenant](https://www.contributor-covenant.org/version/3/0/code_of_conduct/). Unacceptable behavior, harassment, or personal attacks will not be tolerated.

## Reporting Issues

Before opening an issue, please check whether a similar issue already exists.

When creating a new issue, include as much relevant information as possible:

- A clear description of the problem or requested enhancement
- Steps to reproduce the issue, preferably with a minimal reproduction
- Expected behavior and actual behavior
- Environment details, such as OS, Python version, PyTorch version, torch-supa version, commit ID, driver/runtime version, and hardware model
- Relevant code snippets, logs, stack traces, or screenshots

For security vulnerabilities, follow the reporting process in [Security Policy.md](Security%20Policy.md).

## Submitting Code

1. **Fork** the repository to your own account.
2. **Create a feature branch** from the target branch:

```bash
git checkout -b feature/your-feature-name
```

3. **Make your changes** following the style and conventions of the surrounding code.
4. **Add or update tests** for new features, bug fixes, and behavior changes where feasible.
5. **Run relevant tests** before submitting. For example:

```bash
cd torch-supa/test && python3 start_test.py test_abs_kernel.py
```

6. **Submit your changes**:

```bash
git commit -m "fix: resolve issue #123"
```

7. **Push** your branch:

```bash
git push origin feature/your-feature-name
```

8. **Open a Pull Request** against the target branch of this repository.

## Contributor License Agreement (CLA)

This project requires every contributors to sign a Contributor License Agreement (CLA) before their contributions can be merged. By signing the CLA, you grant the project the right to use, modify, and distribute your contribution under the project's open source license, and you confirm that you are the original author or have the rights to submit it.

If you are contributing to this project as an individual, independent of any employer or third‑party institutional work (i.e., the contribution is not within the scope of your employment or any other entity), please sign the Individual CLA. If you are contributing on behalf of your employer, company, government agency, or other legal entity, or if the contribution constitutes work made for hire or falls within your job duties, you must sign the Corporate CLA, which must be executed by an authorized representative of your entity with an official seal (or equivalent electronic authentication). Once your CLA is on file, you may submit pull requests without any additional action. If your CLA is not yet signed, the project maintainers may ask you to sign it before reviewing your contribution.

When creating a pull request, please include a statement in the pull request description confirming that you have read and agreed to the CLA (e.g., “I have signed the CLA and agree to its terms.”). The project may also use an automated bot to verify CLA status.

If you have any questions about the CLA, please contact github@birentech.com.

## Testing Requirements

When adding new features or fixing bugs, please add corresponding tests where feasible. At minimum, run the tests related to the files or functionality you changed. If a test cannot be added or cannot be run locally, explain the reason in the Pull Request description.

## Pull Request Guidelines

A good Pull Request should include:

- A concise summary of the change
- The motivation or issue being addressed
- The testing performed and the results
- Any known limitations, risks, or follow-up work
- Links to related issues, design notes, or discussions, if applicable

For major changes, please open an issue first to discuss the proposed design and expected impact.

## Pull Request Review

- Pull Requests require maintainer review before merging.
- Reviewers may request changes for correctness, maintainability, compatibility, documentation, tests, or security.
- Pull Requests with no activity for an extended period may be closed. They can be reopened when work resumes.

## License

By contributing, you agree that your contributions will be licensed under the open source license specified in the LICENSE file of this repository. If you have any questions about the license, please contact the maintainers.

## Contact

For questions, please email [opensource@birentech.com](mailto:opensource@birentech.com) or use the repository's normal issue or discussion process.

Thank you for your contribution!
