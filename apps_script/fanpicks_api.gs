/** @OnlyCurrentDoc */   // asks Google for this one spreadsheet only, not all of the owner's Sheets

/**
 * Fan Picks API -- a small read-only web app on the "Fan Picks drop box" response sheet.
 *
 * Why it exists: the page (docs/fan.html) cannot read the response sheet itself (Google sends no CORS
 * header on the sheet's CSV), and the repo only sees submissions at the daily pull. This script runs
 * inside the owner's Google account, next to the sheet, and answers the two questions the page needs
 * answered live:
 *
 *   ?a=mine&f=<fan_id>&s=<season>&w=<week>
 *       the latest set this fan locked in for that week -> {ok, code, t, n}  (code: null if none)
 *       f is the fingerprint the page makes from username + PIN. It is not published anywhere public
 *       (grade.json shows a separate public id), so knowing a fan's name is not enough to read his
 *       picks. A PIN is only 4+ digits, though: this keeps casual snooping out, not a determined one.
 *
 *   ?a=who&s=<season>&w=<week>
 *       who is in this week -> {ok, entries: [{fan, arrows, t}]}   names and arrow counts only
 *
 * It only reads. It never writes to the sheet, and it returns no email addresses (those go to a
 * separate private form this script cannot see).
 *
 * Setup and updates: apps_script/README.md. After editing this file, redeploy as a NEW VERSION of the
 * same deployment (Deploy > Manage deployments > edit > Version: New version), or the URL changes.
 */

function doGet(e) {
  var p = (e && e.parameter) || {};
  var out;
  try {
    if (p.a === "mine") out = mine_(p);
    else if (p.a === "who") out = who_(p);
    else out = {ok: false, error: "unknown action"};
  } catch (err) {
    out = {ok: false, error: String(err)};
  }
  return ContentService.createTextOutput(JSON.stringify(out)).setMimeType(ContentService.MimeType.JSON);
}

/** Every response row as {user, code}. The response sheet is the first tab of the spreadsheet. */
function rows_() {
  var sh = SpreadsheetApp.getActiveSpreadsheet().getSheets()[0];
  var v = sh.getDataRange().getValues();
  if (v.length < 2) return [];
  var head = v[0].map(function (h) { return String(h).toLowerCase(); });
  var iCode = head.findIndex(function (h) { return h.indexOf("code") >= 0; });
  var iUser = head.findIndex(function (h) { return h.indexOf("user") >= 0 || h.indexOf("name") >= 0; });
  if (iCode < 0) throw new Error("no Code column in the response sheet");
  return v.slice(1).map(function (r) { return {user: iUser >= 0 ? String(r[iUser] || "") : "", code: String(r[iCode] || "").trim()}; });
}

/** "FAN1.<season>.<week>.<base64url json>" -> {season, week, j}, or null. */
function decode_(code) {
  var m = /^FAN1\.(\d{4})\.(\d{1,2})\.([A-Za-z0-9_\-=]+)$/.exec(code);
  if (!m) return null;
  var b = m[3].replace(/-/g, "+").replace(/_/g, "/").replace(/=+$/, "");
  while (b.length % 4) b += "=";
  try {
    var j = JSON.parse(Utilities.newBlob(Utilities.base64Decode(b)).getDataAsString("UTF-8"));
    return {season: +m[1], week: +m[2], j: j};
  } catch (err) {
    return null;
  }
}

function mine_(p) {
  var f = String(p.f || ""), s = +p.s, w = +p.w;
  if (!/^[0-9a-z]{8,40}$/.test(f) || !s || !w) return {ok: false, error: "bad request"};
  var best = null;
  rows_().forEach(function (r) {
    var x = decode_(r.code);
    if (!x || x.season !== s || x.week !== w || String(x.j.u || "") !== f) return;
    var t = String(x.j.t || "");
    if (!best || t > best.t) best = {code: r.code, t: t, n: (x.j.a || []).length};
  });
  return best ? {ok: true, code: best.code, t: best.t, n: best.n} : {ok: true, code: null};
}

function who_(p) {
  var s = +p.s, w = +p.w, seen = {};
  if (!s || !w) return {ok: false, error: "bad request"};
  rows_().forEach(function (r) {
    var x = decode_(r.code);
    if (!x || x.season !== s || x.week !== w) return;
    var n = (x.j.a || []).length;
    if (!n) return;
    var key = String(x.j.u || x.j.n || r.user), t = String(x.j.t || "");
    if (!seen[key] || t > seen[key].t) seen[key] = {fan: String(x.j.n || r.user || "anonymous").slice(0, 24), arrows: n, t: t};
  });
  return {ok: true, entries: Object.keys(seen).map(function (k) { return seen[k]; })};
}
