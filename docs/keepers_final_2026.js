// FINAL 2026 keeper selections — the league's ACTUAL locked keepers, all 12 teams,
// exported from the browser's Mock Keepers state on 2026-08-25 after selections closed.
// This file is the source of truth: index.html re-applies it to the draft room on every
// page load, so the list survives cleared browser data, other devices, and stray
// "Clear all" clicks. To genuinely change a keeper, edit THIS file — browser edits to
// the kept list only last until the next reload.
// t = index into KEEPERS_2026.teams (0=I Dowdle win … 11=Paul's Perfect Team),
// price = escalated keeper salary charged against that team's $200.
const KEEPERS_FINAL = {
 "version": "2026-08-25",
 "keepers": {
  "Wan'Dale Robinson|WR|NYG":   {"t": 0, "price": 6},
  "Drake London|WR|ATL":        {"t": 0, "price": 42},
  "Rico Dowdle|RB|CAR":         {"t": 0, "price": 6},
  "Chris Olave|WR|NO":          {"t": 1, "price": 20},
  "Jaxson Dart|QB|NYG":         {"t": 1, "price": 6},
  "De'Von Achane|RB|MIA":       {"t": 2, "price": 14},
  "Jahmyr Gibbs|RB|DET":        {"t": 3, "price": 45},
  "Drake Maye|QB|NE":           {"t": 3, "price": 6},
  "Trevor Lawrence|QB|JAX":     {"t": 3, "price": 6},
  "Ja'Marr Chase|WR|CIN":       {"t": 4, "price": 61},
  "Joe Burrow|QB|CIN":          {"t": 4, "price": 6},
  "Christian McCaffrey|RB|SF":  {"t": 4, "price": 54},
  "Puka Nacua|WR|LA":           {"t": 5, "price": 16},
  "Javonte Williams|RB|DAL":    {"t": 5, "price": 18},
  "Brock Bowers|TE|LV":         {"t": 5, "price": 19},
  "Caleb Williams|QB|CHI":      {"t": 6, "price": 25},
  "Kyle Pitts|TE|ATL":          {"t": 6, "price": 4},
  "Trey McBride|TE|ARI":        {"t": 7, "price": 9},
  "Kyren Williams|RB|LA":       {"t": 7, "price": 12},
  "Jameson Williams|WR|DET":    {"t": 7, "price": 9},
  "Bo Nix|QB|DEN":              {"t": 8, "price": 12},
  "Zay Flowers|WR|BAL":         {"t": 8, "price": 16},
  "Breece Hall|RB|NYJ":         {"t": 8, "price": 22},
  "Jonathan Taylor|RB|IND":     {"t": 9, "price": 34},
  "George Pickens|WR|DAL":      {"t": 9, "price": 12},
  "Tetairoa McMillan|WR|CAR":   {"t": 9, "price": 24},
  "Jaxon Smith-Njigba|WR|SEA":  {"t": 10, "price": 13},
  "Chase Brown|RB|CIN":         {"t": 10, "price": 16},
  "Amon-Ra St. Brown|WR|DET":   {"t": 11, "price": 53},
  "James Cook|RB|BUF":          {"t": 11, "price": 26},
  "Travis Etienne|RB|JAX":      {"t": 11, "price": 8}
 }
};
