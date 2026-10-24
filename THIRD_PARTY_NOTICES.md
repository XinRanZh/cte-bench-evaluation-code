# Third-Party Notices

Three of the six service implementations port algorithms from permissively
licensed open-source projects. The ported logic appears in the module code
and in the `SOURCE_CODE` string of each module, and therefore also in the
dataset field `patched_source` of `ratelimiter`, `filesystem`, and
`reservation` scenarios (released in the Hugging Face dataset
`zhangxr7/cte-bench-core-v1`). Those portions remain subject to the upstream
licenses below, whose full texts are reproduced verbatim in
`THIRD_PARTY_LICENSES/`. The `auth`, `bank`, and `cart` services, the
scripts, and all other benchmark code are original and licensed under the
MIT License in `LICENSE`.

| benchmark file | upstream project and source files | upstream license | upstream copyright notice | full license text |
|---|---|---|---|---|
| `testbeds/ratelimiter.py` | python-limits, <https://github.com/alisaifee/limits> (`limits/storage/memory.py`, `limits/strategies.py`; ported from the master branch) | MIT | Copyright (c) 2023 Ali-Akber Saifee | `THIRD_PARTY_LICENSES/python-limits-LICENSE.txt` |
| `testbeds/filesystem.py` | PyFilesystem2, <https://github.com/PyFilesystem/pyfilesystem2> (`fs/memoryfs.py`; ported from the master branch) | MIT | Copyright (c) 2017-2021 The PyFilesystem2 contributors; Copyright (c) 2016-2019 Will McGugan | `THIRD_PARTY_LICENSES/pyfilesystem2-LICENSE.txt` |
| `testbeds/reservation.py` | django-oscar, <https://github.com/django-oscar/django-oscar> (`src/oscar/apps/basket/abstract_models.py`; ported from the master branch) | BSD 3-Clause | Copyright (c) 2011-present Tangent Communications PLC and individual contributors. All rights reserved. | `THIRD_PARTY_LICENSES/django-oscar-LICENSE.txt` |

The MIT notices require that the copyright notice and permission notice be
included in all copies or substantial portions of the software; they are
included in the files listed above. Both MIT licenses state that the
software is provided "as is", without warranty of any kind.

The module headers of the three ported files state the upstream license and
that the port is intended for research benchmarking. That phrase describes
the intended use only; it adds no restriction to the MIT and BSD terms,
which govern use and redistribution.

django-oscar is distributed under the BSD 3-Clause License. Redistributions
of the derived code in this repository and in the dataset retain the
copyright notice, the list of conditions, and the disclaimer reproduced in
`THIRD_PARTY_LICENSES/django-oscar-LICENSE.txt`. In accordance with its third
condition, neither the name of Tangent Communications PLC nor the names of
its contributors are used to endorse or promote this benchmark, which is not
affiliated with or endorsed by them. The upstream disclaimer applies: the
software is provided by the copyright holders and contributors "as is",
without any express or implied warranties.

Provenance of the license texts: retrieved on 2026-10-21 from the upstream
master branches (`LICENSE.txt` of alisaifee/limits, git blob
`2261d327ad9be9232d7d08158db2ccae582b98e6`; `LICENSE` of
PyFilesystem/pyfilesystem2, git blob
`34845692b8b5ffeb8ec684f1abdd775e7a6bd179`; `LICENSE` of
django-oscar/django-oscar, git blob
`614855e086d91bb97280e4f6330b30ff9c64fb85`).
