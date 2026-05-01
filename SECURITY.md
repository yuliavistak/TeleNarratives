# Security Guidelines

## Environment Variables and Secrets

This project uses environment variables for all sensitive configuration. **Never commit secrets to version control.**

### Required Setup

1. **Copy the example file:**
   ```bash
   cp .env.example .env
   ```

2. **Fill in your credentials in `.env`:**
   - Telegram API credentials (from https://my.telegram.org)
   - Neo4j database credentials

3. **Verify `.env` is gitignored:**
   ```bash
   git check-ignore .env
   ```
   Should output: `.env`

### Security Best Practices

#### ✅ DO:
- Store all secrets in `.env` file
- Keep `.env` in `.gitignore` (already configured)
- Use `.env.example` as a template (without real values)
- Load environment variables at runtime, never hardcode
- Use `python-dotenv` to load `.env` files
- Rotate credentials if they are ever exposed

#### ❌ DON'T:
- Commit `.env` files to git
- Hardcode API keys, passwords, or tokens in source code
- Share `.env` files via email, chat, or other insecure channels
- Commit credentials in comments or documentation
- Use production credentials in development/testing

### Environment Variable Loading

The project automatically loads environment variables from:
1. `.env` file in project root (recommended)
2. System environment variables (for CI/CD, containers, etc.)

Priority order:
1. System environment variables (highest priority)
2. `.env` file in current working directory
3. `.env` file in project root

### Sensitive Data Checklist

The following should **always** be in `.env`, never in code:
- ✅ `TELEGRAM_API_ID` - Telegram API identifier
- ✅ `TELEGRAM_API_HASH` - Telegram API secret hash
- ✅ `NEO4J_URI` - Neo4j connection URI
- ✅ `NEO4J_PASSWORD` - Neo4j database password
- ✅ `NEO4J_USERNAME` - Neo4j username (if not default)

### Verification

To verify your setup is secure:

```bash
# Check that .env is ignored
git status | grep .env
# Should show nothing (file is ignored)

# Check for hardcoded secrets in code
grep -r "api_id.*=" src/ --exclude-dir=__pycache__
grep -r "api_hash.*=" src/ --exclude-dir=__pycache__
grep -r "password.*=" src/ --exclude-dir=__pycache__
# Should only show variable assignments, not actual values

# Verify .env.example has no real credentials
grep -E "(your_|example|placeholder)" .env.example
# Should show placeholder values only
```

### If Credentials Are Exposed

If you accidentally commit credentials:

1. **Immediately rotate/revoke the exposed credentials:**
   - Telegram: Create new API credentials at https://my.telegram.org
   - Neo4j: Change password in Neo4j dashboard

2. **Remove from git history:**
   ```bash
   # Use git-filter-repo or BFG Repo-Cleaner
   # This removes the file from all commits
   ```

3. **Update `.env` with new credentials**

4. **Notify team members** if working in a team

### CI/CD Integration

For CI/CD pipelines, use secure secret management:
- GitHub Actions: Use Secrets
- GitLab CI: Use CI/CD Variables (masked)
- Jenkins: Use Credentials plugin
- Docker: Use secrets or environment variables (not in Dockerfile)

Example for GitHub Actions:
```yaml
env:
  TELEGRAM_API_ID: ${{ secrets.TELEGRAM_API_ID }}
  TELEGRAM_API_HASH: ${{ secrets.TELEGRAM_API_HASH }}
```

### Additional Resources

- [OWASP Secrets Management](https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html)
- [12-Factor App: Config](https://12factor.net/config)
- [Python-dotenv Documentation](https://github.com/theskumar/python-dotenv)

