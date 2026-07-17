# Information security policy

## Access control

Access is granted on the principle of least privilege and reviewed quarterly.
Privileged accounts require hardware second-factor authentication. Shared
accounts are prohibited without a documented exception approved by the security
officer, and every exception expires after ninety days.

## Credentials

Secrets are never committed to source control. Application credentials are
issued from the secret manager and rotated automatically every thirty days.
A credential believed to be exposed must be rotated within one hour of the
report, regardless of the hour or day.

## Data retention

Customer records are retained for seven years after the end of the contractual
relationship, then deleted. Operational logs are retained for eighteen months.
Backups follow the retention of the data they contain; there is no separate
backup retention schedule.

## Incident response

A suspected incident is reported to the security officer immediately and
triaged within one hour. Incidents rated high or critical require notification
of affected customers within seventy-two hours of confirmation. The postmortem
is blameless and published internally within ten working days.

## Third parties

Vendors handling customer data undergo review before onboarding and annually
thereafter. The review covers their subprocessors, their breach history, and
the location of the data. A vendor without a current review may not be granted
production access.
