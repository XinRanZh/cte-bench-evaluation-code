from . import bank, cart, auth, reservation, ratelimiter, filesystem
TESTBEDS = {
    "bank": bank,
    "cart": cart,
    "auth": auth,
    "reservation": reservation,
    "ratelimiter": ratelimiter,
    "filesystem": filesystem,
}
