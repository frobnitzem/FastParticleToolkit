Fast Particle Toolkit (FPT)
===========================

[![License FPT](https://img.shields.io/badge/license-GPLv3-blue.svg?label=FPT)](https://www.gnu.org/licenses/gpl-3.0.html)

Introduction
------------

Fast Particle Toolkit is a Swiss-Army Knife for computations involving
particles.  It does a few things well:

- Compute one-body functions, f(i) = g(ri; ni)
- Compute two-body functions, f(i) = sum(neighbors of i) g(ri, rj; ni, nj)
- Aggregate per-particle data, A(x) = sum(i) f(x, ri; ni)

This code implements the summations.  Your code implements the functions.

Attribution
-----------

FPT is a *scientific project*. If you **present and/or publish** scientific
results that used FPT, you must **reference** this work to show your support.

Our according **up-to-date publication** at **the time of your publication**
should be inquired from:
- [REFERENCE.md](https://raw.githubusercontent.com/frobnitzem/FPT/master/REFERENCE.md)

Oral Presentations
------------------

The following logo should be added to all figures in **oral presentations**
that were generated with the help of FPT.

(*coming soon*) FPT-logo.svg, FPT-logo.png

Software License
----------------

*FPT* is licensed under the **GPLv3+**.
It makes heavy use of modern accelerator backend libraries:
 * https://github.com/alpaka-group/mallocMC
 * https://github.com/alpaka-group/alpaka
 * https://github.com/alpaka-group/cupla

For a detailed description, please refer to [LICENSE.md](LICENSE.md)

********************************************************************************

Install
-------

This repository is meant to be included as a submodule in your project.

Quick-Start Instructions:
```
mkdir MyProject
cd MyProject
git init
git submodule add git@github.com:frobnitzem/FastParticleToolkit.git
FastParticleToolkit/boostrap.sh "myProg" # name of your compiled executable

git commit -am "Initial Project"
FastParticleToolkit/build.sh ../install_dir # test build
```

For notes on navigating the build process with your hardware,
see [INSTALL.rst](INSTALL.rst).

Users
-----

Dear User, please be aware that this is an **open beta release**!
We hereby emphasize that we are still actively developing FPT at great
speed and do, from time to time, break backwards compatibility.

When using this software, please stick to the `master` branch containing the
latest *stable* release. It also contains a file `CHANGELOG.md` with the
latest changes (and how to update your simulations). Read it first before
updating between two versions! Also, we add a git `tag` according to a version
number for each release in `master`.

For any questions regarding the usage of FPT just contact the
developers and maintainers directly.

Before you post a question, browse the FPT
[documentation](https://github.com/frobnitzem/FastParticleToolkit/FastParticleToolkit/search?l=markdown), and
[issue tracker](https://github.com/frobnitzem/FastParticleToolkit/issues)
to see if your question has been answered already.

FPT is a collaborative project.
We thus encourage users to engage in answering questions of other users and post solutions to problems to the list.
A problem you have encountered might be the future problem of another user.

In addition, please consider using the collaborative features of GitHub if you have questions or comments on code or documentation.
This will allow other users to see the piece of code or documentation you are referring to.

Main ressources are in our [online manual - coming soon](https://frobnitzem.github.io/FastParticleToolkit), built from 
[`.rst` (reStructuredText)](http://www.sphinx-doc.org/en/stable/rest.html) files in this repository,
and other [`.md` (Markdown)](http://commonmark.org/help/) docs here.

Software Upgrades
-----------------

FPT follows a
[master - dev](http://nvie.com/posts/a-successful-git-branching-model/)
development model. That means our latest stable release is shipped in a branch
called `master` while new and frequent changes to the code are incooporated
in the development branch `dev`.

Every time we update the *master* branch, we publish a new release
of FPT. Before you pull the changes in, please read our
[ChangeLog](CHANGELOG.md)!
You may have to update some of your calculation's config files by
hand since FPT is an active project and new features often require changes
in input files. Additionally, a full description of new features and fixed bugs
in comparison to the previous release is provided in that file.

In case you decide to use *new, potentially buggy and experimental* features
from our `dev` branch, be aware that support is very limited and you must
participate or at least follow the development yourself. Syntax changes
and in-development bugs will *not* be announced outside of their according pull
requests and issues.

Before drafting a new release, we open a new `release-*` branch from `dev` with
the `*` being the version number of the upcoming release. This branch only
receives bug fixes (feature freeze) and users are welcome to try it out
(however, the change log and a detailed announcement might still be missing in
it).

Developers
----------

### How to participate

See [CONTRIBUTING.md](CONTRIBUTING.md)

Active Team
-----------

### Maintainers* and core developers

- Dr. David M. Rogers

********************************************************************************
