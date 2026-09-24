# Fan Picks API — setup (about 15 minutes, once)

`fanpicks_api.gs` is a small read-only web app that runs in your Google account next to the
**Fan Picks drop box** response sheet. It lets the page:

- **find a fan's picks on any device.** Type username + PIN on a new phone and the picks you
  locked in elsewhere appear.
- **show "who's in" live.** Until now that came from a file baked once a day.

The page already knows how to use it. It stays switched off until the script's address is in
`docs/fan/dropbox.json` under `"api"`.

## 1. Open the script editor on the response sheet
1. Open the response sheet — the one with the **Timestamp / Username / Code** columns. Its link
   is `sheet_edit` in `docs/fan/dropbox.json`.
2. Menu: **Extensions → Apps Script**. A new tab opens with a project attached to this sheet.

## 2. Paste the code
1. In the editor, select everything in `Code.gs` and delete it.
2. Paste the whole contents of `apps_script/fanpicks_api.gs` from this repo.
3. Click the project name ("Untitled project") and rename it **Fan Picks API**.
4. Save (the disk icon, or Ctrl/Cmd + S).

## 3. Deploy it as a web app
1. Top right: **Deploy → New deployment**.
2. Click the gear next to "Select type" and choose **Web app**.
3. Fill in:
   - Description: `v1`
   - **Execute as: Me**
   - **Who has access: Anyone**
4. Click **Deploy**.

## 4. Authorize it (first time only)
1. **Authorize access** → pick your Google account.
2. Google shows **"Google hasn't verified this app."** That's normal for a script you wrote
   yourself. Click **Advanced → Go to Fan Picks API (unsafe)**.
3. The permission it asks for is to see and edit **this one spreadsheet** (the script declares
   `@OnlyCurrentDoc`). It doesn't ask for your other files, and the code only ever reads.
4. Click **Allow**.

## 5. Copy the address and test it
1. Copy the **Web app URL**. It ends in `/exec`.
2. Paste it into a browser tab with this added to the end: `?a=who&s=2026&w=3`
   You should see something like `{"ok":true,"entries":[{"fan":"Rob_Burke","arrows":5,...}]}`.
3. Send the URL over (or put it in `docs/fan/dropbox.json` as `"api": "<url>"`) and the page
   switches it on.

## Changing the script later
Edit, save, then **Deploy → Manage deployments → pencil icon → Version: New version → Deploy**.
That keeps the same URL. "New deployment" would make a new one, and the page would still be
pointing at the old URL.

## What it exposes
- `?a=mine&f=<fingerprint>&s=&w=` returns one fan's latest pick code for that week. The
  fingerprint comes from username + PIN, and since 2026-09-24 it isn't published anywhere public
  (`grade.json` shows a separate public id instead). Knowing a fan's name isn't enough.
  PINs are only 4+ digits, though, so this keeps casual snooping out, not a determined person.
- `?a=who&s=&w=` returns display names and arrow counts: no fingerprints, codes or emails.
- Nothing writes to the sheet. Email addresses live in a separate private form the script can't see.

**Still open:** the response sheet itself is shared read-only by link (the daily Actions pull
reads it that way), and that link is in the page's public config. The sheet holds every code,
fingerprints included. Closing it means the Actions pull reading through this script with a
secret token instead, then turning off link sharing.
