# Third-Party Notices

Three compact service implementations port core algorithmic behavior from
permissively licensed open-source projects:

- `testbeds/reservation.py`: django-oscar style availability/reservation
  behavior, upstream BSD 3-Clause.
- `testbeds/ratelimiter.py`: python-limits style fixed-window/rate-limit
  behavior, upstream MIT.
- `testbeds/filesystem.py`: pyfilesystem2-style in-memory filesystem
  behavior, upstream MIT.

`auth`, `bank`, and `cart` are synthetic deterministic services. The
released benchmark data are licensed CC-BY-4.0; benchmark code is intended
for MIT-style release. Replace this notice with full project license files
before camera-ready public hosting if your institution requires a specific
license template.
