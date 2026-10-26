# Repository Rename

On 2026-10-26, the local project repository and workspace directory were renamed from `yizhu` to `zdog-yizhu`.

## Paths

- Repository root: `/home/zhang/workspace/zdog-yizhu`
- Training-data storage: `/data/zdog-yizhu/server-cache/training`
- Repository-compatible training path: `server/.cache/training`

The product name and public domains remain unchanged: “一株”, `yizhu-h.942295.xyz`, and `yizhuh5-h.942295.xyz`. Python import names and existing command names such as `yizhu-server` also remain stable to avoid an unnecessary application-level migration.

## Updated host configuration

The rename updated and verified:

- the persistent training-data bind mount in `/etc/fstab`;
- the user systemd service working directory and executable path;
- the Nginx H5 document root;
- the SELinux `httpd_sys_content_t` file-context rule;
- project scripts, documentation, manifests, virtual-environment entry points, and package metadata.

The client package is now named `zdog-yizhu`; the Python distribution is `zdog-yizhu-server`. The production model classifier was not changed. Its metadata checksum was regenerated because recorded absolute artifact paths changed during relocation.

## Verification

```sh
findmnt --target /home/zhang/workspace/zdog-yizhu/server/.cache/training
systemctl --user status yizhu-server
sudo nginx -t
curl https://yizhu-h.942295.xyz/api/health
```

The old workspace and training-data directories should not exist. `/etc/fstab` and Nginx backup files created during migration are retained outside the repository for host-level rollback.
