# Thin aliases. Everything real lives in src/check.py.
.PHONY: check check-write
check:                ## every CPU-only check (login-node safe, ~12 s)
	@./src/check.py
check-write:          ## refresh the goldens -- review the diff before committing
	@./src/check.py --write
