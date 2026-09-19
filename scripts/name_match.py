"""Reconciling player names between the FFA file and the DraftKings props feed.

DraftKings writes "Kenny Gainwell" where the FFA file writes "Kenneth Gainwell", and
"Joshua Palmer" where FFA writes "Josh Palmer". sabersim_weekly joins props on the
normalized full name, which folds case, punctuation and Jr/Sr/III but cannot bridge a
nickname, so those players' lines are dropped. That costs the market blend — and worse,
the resulting `no_line` flag feeds the DOUBT haircut, whose premise is "the books will
not price him, so he is probably not playing". A spelling mismatch inverts precisely the
evidence that rule reads.

Surname alone is not safe to match on. Week 2 by itself had DK pricing a Cameron Latu
against FFA's Keleki Latu, a Dohnte Meyers against Jakobi Meyers, and a Ke'Shawn Williams
against seven other Williamses. A match therefore needs all four of: same surname, same
game, first names that look like one person written two ways, and exactly one candidate.

Names reaching these functions are already normalized (lowercased, punctuation and
generational suffixes stripped).
"""


def same_person(a, b):
    """True when two first names are plausibly one person written two ways.

    Accepts josh/joshua and kenny/kenneth (four or more shared leading characters) and a
    short form wholly contained in the long one (cam/cameron, dan/daniel). Deliberately
    strict: a miss only forfeits the market blend, which is the behaviour we already had,
    whereas a false match would attach one player's betting line to another. Pairs it
    cannot see through, such as mike/michael, stay unmatched and are reported so a human
    can look rather than being guessed at.
    """
    if a == b:
        return True
    shared = 0
    for x, y in zip(a, b):
        if x != y:
            break
        shared += 1
    if shared >= 4:
        return True
    return shared >= 3 and shared in (len(a), len(b))


def reconcile(dk_names, event_teams, players):
    """Map unmatched DK names onto slate players.

    dk_names    iterable of (event_id, normalized DK name)
    event_teams {event_id: set of the two team codes playing in it}
    players     iterable of (normalized slate player name, team code)

    Returns (alias, ambiguous): alias maps (event_id, DK name) to the slate name it means,
    and ambiguous lists DK names that had more than one plausible match and were left alone.

    The key carries the event on purpose. Two different players can share a DK spelling in
    two different games — a Chris Smith in each, written Christian Smith in one FFA row and
    Christopher Smith in the other. Keying on the name alone would let the first game's
    mapping win and then be applied to every row of that name, which the caller groups by
    name, merging one player's lines into the other. That is the mis-attribution this whole
    module exists to avoid, so the same-game guard has to survive into how the alias is
    applied, not just how it is chosen.
    """
    known = {n for n, _ in players}
    by_team_surname = {}
    for n, team in players:
        parts = n.split()
        if len(parts) >= 2:
            by_team_surname.setdefault((team, parts[-1]), []).append(n)

    alias, ambiguous = {}, set()
    for event_id, dk in dk_names:
        if dk in known or (event_id, dk) in alias:
            continue
        parts = dk.split()
        if len(parts) < 2:
            continue
        first, surname = parts[0], parts[-1]
        cand = {other for team in event_teams.get(event_id, ())
                for other in by_team_surname.get((team, surname), [])
                if same_person(first, other.split()[0])}
        if len(cand) == 1:
            alias[(event_id, dk)] = cand.pop()
        elif cand:
            ambiguous.add(dk)
    return alias, sorted(ambiguous)
