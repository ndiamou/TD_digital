"""Création des éléments de la base ICL Pulse (PostgreSQL).

Exemples :
    python icl_pulse_db.py insert t_project project_num=P000001 project_name="Test" vitesse_fixe=true
    python icl_pulse_db.py user --email a@b.com --password secret --sso 123456789 --admin
    python icl_pulse_db.py ipsb --psb-num 1 --level-num 2 --description "Pression aspiration" --unit bar
    python icl_pulse_db.py curve --project-num P000001 --ipsb-num IPSB_1_2_01 --points points.csv
    python icl_pulse_db.py compo-gaz --project-num P000001 --rev 1
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Callable, Iterable

import bcrypt
import psycopg
from dotenv import load_dotenv
from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


def _bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "oui", "o"):
        return True
    if s in ("0", "false", "f", "no", "n", "non"):
        return False
    raise ValueError(f"Booléen invalide : {v!r}")


def _date(v: Any) -> date:
    return v if isinstance(v, date) else date.fromisoformat(str(v))


def _ts(v: Any) -> datetime:
    return v if isinstance(v, datetime) else datetime.fromisoformat(str(v))


def _json(v: Any) -> Jsonb:
    return Jsonb(json.loads(v) if isinstance(v, str) else v)


def _bytes(v: Any) -> bytes:
    return v if isinstance(v, bytes) else bytes.fromhex(str(v))


S, I, N, B, D, T, J, X = str, int, Decimal, _bool, _date, _ts, _json, _bytes

# Colonnes insérables par table (hors colonnes auto-générées SERIAL / DEFAULT).
SCHEMA: dict[str, dict[str, Callable[[Any], Any]]] = {
    "t_user": {"email": S, "is_active": B, "is_admin": B, "sso": S, "hashed_password": S},
    "t_project": {"project_num": S, "project_name": S, "customer_name": S, "sales_date": D,
                  "sso_tester": S, "vitesse_fixe": B, "vitesse_rot": I},
    "t_technical_data": {"project_num": S, "section_num": I, "notes": S, "rotor_nb": I,
                         "d2_std_1st_imp": N, "d2_mean": N, "volumeflow_d0": N, "volumeflow_d": N,
                         "volumeflow_l1": N, "volumeflow_l2": N, "elt_deprimogene": S, "k_factor": N},
    "t_essais": {"project_num": S, "type_curve": S, "mach": N, "notes": S},
    "t_psb": {"psb_name": S, "psb_description": S, "remarks": S},
    "t_psb_level": {"level_num": I, "psb_num": I, "level_name": S, "level_description": S, "remarks": S},
    "t_ipsb": {"ipsb_num": S, "psb_num": I, "level_num": I, "ipsb_description": S, "std_unit": S,
               "curve_type": S, "notes": S, "color": S},
    "t_ipsb_project": {"project_num": S, "ipsb_num": S, "tag_num": X, "notes": S},
    "t_compound": {"id": I, "name": S, "formulation": S, "casrn": S},
    "t_compo_gaz": {"project_num": S, "rev": I, "notes": S, "creation_date": T, "equation_etat": S},
    "t_compo_gaz_set": {"compo_gaz": I, "num_compo": I, "notes": S},
    "t_compo_gaz_detail": {"compo_gaz_set": I, "compound_id": I, "tauxmol": N},
    "t_graph": {"graph_description": S, "notes": S, "sso_owner": S, "creation_date": T},
    "t_curve": {"project_num": S, "ipsb_num": S, "unit": S, "delta_y": N, "link_type": S, "notes": S,
                "revision": I, "creation_date": T, "sso_owner": S, "isref": B, "is_comp_res": B},
    "t_curve_points": {"curve_id": I, "x": N, "y": N},
    "t_curve_timestamp": {"curve_id": I, "tmsp": T, "y": N, "qual_data": B},
    "t_graph_curves": {"graph_id": I, "ipsb_id": I},
    "t_alarms": {"project_num": S, "tmsp": T, "alarm_num": S, "alarm_description": S, "notes": S},
    "t_essais_polynome": {"essais_id": I, "phi_surge": N, "phi_shoke": N, "degre0": N, "degre1": N,
                          "degre2": N, "degre3": N, "degre4": N},
    "t_essais_point": {"essais_id": I, "curve_id": I, "phi_surge": N, "phi_shoke": N},
    "t_computation_history": {"project_num": S, "computation_history": J, "computation_curve": J,
                              "computation_result": J, "creation_date": T, "sso": S,
                              "computation_curve_result_id": J},
}


# --------------------------------------------------------------------------- connexion

def connect(env_file: str = ".env") -> psycopg.Connection:
    load_dotenv(env_file)
    # POSTGRES_SERVER peut contenir "hote:port/base?" : on n'en garde que l'hôte.
    server = os.environ["POSTGRES_SERVER"]
    host = server.split("/", 1)[0].split(":", 1)[0]
    return psycopg.connect(
        host=host,
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        dbname=os.environ["POSTGRES_DB"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        sslmode=os.getenv("POSTGRES_SSLMODE", "prefer"),
        row_factory=dict_row,
    )


# --------------------------------------------------------------------------- générique

def _convert(table: str, values: dict[str, Any]) -> dict[str, Any]:
    if table not in SCHEMA:
        raise ValueError(f"Table inconnue : {table}")
    cols = SCHEMA[table]
    unknown = set(values) - set(cols)
    if unknown:
        raise ValueError(f"Colonnes inconnues pour {table} : {', '.join(sorted(unknown))}")
    return {k: (None if v is None else cols[k](v)) for k, v in values.items()}


def insert(conn: psycopg.Connection, table: str, values: dict[str, Any]) -> dict[str, Any]:
    """Insère une ligne et retourne la ligne créée."""
    data = _convert(table, values)
    if data:
        query = sql.SQL("INSERT INTO {} ({}) VALUES ({}) RETURNING *").format(
            sql.Identifier(table),
            sql.SQL(", ").join(map(sql.Identifier, data)),
            sql.SQL(", ").join(sql.Placeholder() * len(data)),
        )
    else:
        query = sql.SQL("INSERT INTO {} DEFAULT VALUES RETURNING *").format(sql.Identifier(table))
    with conn.cursor() as cur:
        cur.execute(query, list(data.values()))
        return cur.fetchone()


def insert_many(conn: psycopg.Connection, table: str, rows: Iterable[dict[str, Any]]) -> int:
    rows = [_convert(table, r) for r in rows]
    if not rows:
        return 0
    cols = list(rows[0])
    query = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
        sql.Identifier(table),
        sql.SQL(", ").join(map(sql.Identifier, cols)),
        sql.SQL(", ").join(sql.Placeholder() * len(cols)),
    )
    with conn.cursor() as cur:
        cur.executemany(query, [[r.get(c) for c in cols] for r in rows])
    return len(rows)


# --------------------------------------------------------------------------- métiers

def create_user(conn, email: str, password: str, sso: str | None = None,
                is_admin: bool = False, is_active: bool = True) -> dict[str, Any]:
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    return insert(conn, "t_user", {"email": email, "hashed_password": hashed, "sso": sso,
                                   "is_admin": is_admin, "is_active": is_active})


def next_ipsb_num(conn, psb_num: int, level_num: int) -> str:
    """Construit IPSB_%PSB_NUM%_%LEVEL_NUM%_XX avec XX = prochain index libre."""
    prefix = f"IPSB_{psb_num}_{level_num}_"
    with conn.cursor() as cur:
        cur.execute("SELECT ipsb_num FROM t_ipsb WHERE ipsb_num LIKE %s", (prefix + "%",))
        used = [int(r["ipsb_num"][len(prefix):]) for r in cur.fetchall()
                if r["ipsb_num"][len(prefix):].isdigit()]
    num = f"{prefix}{max(used, default=0) + 1:02d}"
    if len(num) > 14:
        raise ValueError(f"ipsb_num trop long (14 car. max) : {num}")
    return num


def create_ipsb(conn, psb_num: int, level_num: int, **fields) -> dict[str, Any]:
    fields.setdefault("ipsb_num", next_ipsb_num(conn, psb_num, level_num))
    return insert(conn, "t_ipsb", {"psb_num": psb_num, "level_num": level_num, **fields})


def create_curve(conn, project_num: str, ipsb_num: str,
                 points: Iterable[tuple[Any, Any]] = (),
                 timestamps: Iterable[tuple[Any, Any, Any]] = (), **fields) -> dict[str, Any]:
    fields.setdefault("creation_date", datetime.now())
    curve = insert(conn, "t_curve", {"project_num": project_num, "ipsb_num": ipsb_num, **fields})
    cid = curve["curve_id"]
    insert_many(conn, "t_curve_points", ({"curve_id": cid, "x": x, "y": y} for x, y in points))
    insert_many(conn, "t_curve_timestamp",
                ({"curve_id": cid, "tmsp": t, "y": y, "qual_data": q} for t, y, q in timestamps))
    return curve


def get_or_create_compo_gaz(conn, project_num: str, rev: int, **fields) -> dict[str, Any]:
    """Retourne la révision (project_num, rev) de t_compo_gaz ; la crée si absente,
    sinon met à jour les champs fournis (notes, equation_etat...)."""
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM t_compo_gaz WHERE project_num = %s AND rev = %s ORDER BY id LIMIT 1",
                    (project_num, rev))
        compo = cur.fetchone()
    if compo is None:
        fields.setdefault("creation_date", datetime.now())
        return insert(conn, "t_compo_gaz", {"project_num": project_num, "rev": rev, **fields})
    data = _convert("t_compo_gaz", fields)
    if not data:
        return compo
    query = sql.SQL("UPDATE t_compo_gaz SET {} WHERE id = %s RETURNING *").format(
        sql.SQL(", ").join(sql.SQL("{} = %s").format(sql.Identifier(k)) for k in data))
    with conn.cursor() as cur:
        cur.execute(query, [*data.values(), compo["id"]])
        return cur.fetchone()


def save_compo_gaz_set(conn, compo_gaz_id: int, num_compo: int, details: dict[int, Any],
                       notes: str | None = None, replace: bool = True) -> dict[str, Any]:
    """Crée ou met à jour la composition num_compo d'une révision et son détail.

    details = {compound_id: tauxmol}. Si replace, les composés absents de details sont supprimés.
    """
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO t_compo_gaz_set (compo_gaz, num_compo, notes) VALUES (%s, %s, %s) "
            "ON CONFLICT (compo_gaz, num_compo) DO UPDATE "
            "SET notes = COALESCE(EXCLUDED.notes, t_compo_gaz_set.notes) RETURNING *",
            (compo_gaz_id, num_compo, notes))
        gset = cur.fetchone()
        rows = [_convert("t_compo_gaz_detail", {"compo_gaz_set": gset["id"], "compound_id": c,
                                                "tauxmol": t}) for c, t in details.items()]
        if replace:
            cur.execute("DELETE FROM t_compo_gaz_detail WHERE compo_gaz_set = %s "
                        "AND NOT (compound_id = ANY(%s))",
                        (gset["id"], [r["compound_id"] for r in rows]))
        cur.executemany(
            "INSERT INTO t_compo_gaz_detail (compo_gaz_set, compound_id, tauxmol) VALUES (%s, %s, %s) "
            "ON CONFLICT (compo_gaz_set, compound_id) DO UPDATE SET tauxmol = EXCLUDED.tauxmol",
            [(r["compo_gaz_set"], r["compound_id"], r["tauxmol"]) for r in rows])
    gset["nb_details"] = len(rows)
    return gset


def create_compo_gaz(conn, project_num: str, rev: int, sets: list[dict[str, Any]],
                     **fields) -> dict[str, Any]:
    """Crée ou met à jour une révision et ses compositions.

    sets = [{"num_compo": 1, "notes": "...", "details": {compound_id: tauxmol, ...}}, ...]
    """
    compo = get_or_create_compo_gaz(conn, project_num, rev, **fields)
    compo["sets"] = [save_compo_gaz_set(conn, compo["id"], s["num_compo"], s.get("details", {}),
                                        s.get("notes"), s.get("replace", True)) for s in sets]
    return compo


def get_compo_gaz(conn, project_num: str | None = None, rev: int | None = None,
                  compo_gaz_id: int | None = None) -> list[dict[str, Any]]:
    """Retourne les compositions gaz existantes (filtres optionnels) sous forme imbriquée :
    [{id, project_num, rev, ..., "sets": [{id, num_compo, notes, "details": [...]}]}]
    """
    conds, params = [], []
    for col, val in (("g.project_num", project_num), ("g.rev", rev), ("g.id", compo_gaz_id)):
        if val is not None:
            conds.append(f"{col} = %s")
            params.append(val)
    where = f"WHERE {' AND '.join(conds)}" if conds else ""
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT g.id, g.project_num, g.rev, g.notes, g.creation_date, g.equation_etat,
                   s.id AS set_id, s.num_compo, s.notes AS set_notes,
                   d.compound_id, c.name, c.formulation, c.casrn, d.tauxmol
            FROM t_compo_gaz g
            LEFT JOIN t_compo_gaz_set s ON s.compo_gaz = g.id
            LEFT JOIN t_compo_gaz_detail d ON d.compo_gaz_set = s.id
            LEFT JOIN t_compound c ON c.id = d.compound_id
            {where}
            ORDER BY g.project_num, g.rev, g.id, s.num_compo, d.compound_id""", params)
        rows = cur.fetchall()

    compos: dict[int, dict[str, Any]] = {}
    sets: dict[int, dict[str, Any]] = {}
    for r in rows:
        compo = compos.setdefault(r["id"], {k: r[k] for k in ("id", "project_num", "rev", "notes",
                                                              "creation_date", "equation_etat")}
                                  | {"sets": []})
        if r["set_id"] is None:
            continue
        if r["set_id"] not in sets:
            sets[r["set_id"]] = {"id": r["set_id"], "num_compo": r["num_compo"],
                                 "notes": r["set_notes"], "details": []}
            compo["sets"].append(sets[r["set_id"]])
        if r["compound_id"] is not None:
            sets[r["set_id"]]["details"].append(
                {k: r[k] for k in ("compound_id", "name", "formulation", "casrn", "tauxmol")})
    return list(compos.values())


# --------------------------------------------------------------------------- CLI

def _parse_kv(pairs: list[str]) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"Format attendu colonne=valeur : {p!r}")
        k, v = p.split("=", 1)
        out[k.strip()] = None if v.upper() == "NULL" else v
    return out


def _read_csv(path: str) -> list[list[str]]:
    with open(path, newline="", encoding="utf-8") as f:
        sample = f.read(2048)
        f.seek(0)
        rows = list(csv.reader(f, csv.Sniffer().sniff(sample, delimiters=",;\t")))
    if rows and not _is_number(rows[0][-1]):
        rows = rows[1:]  # en-tête
    return rows


def _is_number(s: str) -> bool:
    try:
        Decimal(s)
        return True
    except Exception:
        return False


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Création d'éléments dans la base ICL Pulse")
    p.add_argument("--env-file", default=".env", help="Fichier d'environnement (défaut : .env)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("insert", help="Insertion générique : TABLE col=valeur ...")
    sp.add_argument("table", choices=sorted(SCHEMA))
    sp.add_argument("values", nargs="*")

    sub.add_parser("tables", help="Liste les tables et colonnes insérables")

    sp = sub.add_parser("user", help="Crée un utilisateur (mot de passe haché bcrypt)")
    sp.add_argument("--email", required=True)
    sp.add_argument("--password", required=True)
    sp.add_argument("--sso")
    sp.add_argument("--admin", action="store_true")
    sp.add_argument("--inactive", action="store_true")

    sp = sub.add_parser("ipsb", help="Crée un IPSB avec numéro auto IPSB_<psb>_<level>_XX")
    sp.add_argument("--psb-num", type=int, required=True)
    sp.add_argument("--level-num", type=int, required=True)
    sp.add_argument("--description")
    sp.add_argument("--unit")
    sp.add_argument("--curve-type")
    sp.add_argument("--notes")
    sp.add_argument("--color")

    sp = sub.add_parser("curve", help="Crée une courbe et ses points depuis un CSV")
    sp.add_argument("--project-num", required=True)
    sp.add_argument("--ipsb-num", required=True)
    sp.add_argument("--points", help="CSV x,y")
    sp.add_argument("--timestamps", help="CSV tmsp,y[,qual_data]")
    sp.add_argument("fields", nargs="*", help="Autres colonnes de t_curve : col=valeur")

    sp = sub.add_parser("compo-gaz", help="Liste les compositions gaz existantes")
    sp.add_argument("--project-num")
    sp.add_argument("--rev", type=int)

    args = p.parse_args(argv)

    if args.cmd == "tables":
        for t, cols in SCHEMA.items():
            print(f"{t}: {', '.join(cols)}")
        return 0

    try:
        with connect(args.env_file) as conn:  # commit auto en sortie, rollback si exception
            if args.cmd == "insert":
                row = insert(conn, args.table, _parse_kv(args.values))
            elif args.cmd == "user":
                row = create_user(conn, args.email, args.password, args.sso,
                                  is_admin=args.admin, is_active=not args.inactive)
                row.pop("hashed_password", None)
            elif args.cmd == "ipsb":
                extra = {"ipsb_description": args.description, "std_unit": args.unit,
                         "curve_type": args.curve_type, "notes": args.notes, "color": args.color}
                row = create_ipsb(conn, args.psb_num, args.level_num,
                                  **{k: v for k, v in extra.items() if v is not None})
            elif args.cmd == "compo-gaz":
                row = get_compo_gaz(conn, args.project_num, args.rev)
            else:
                pts = [(r[0], r[1]) for r in _read_csv(args.points)] if args.points else []
                tss = [(r[0], r[1], r[2] if len(r) > 2 else None)
                       for r in _read_csv(args.timestamps)] if args.timestamps else []
                row = create_curve(conn, args.project_num, args.ipsb_num, pts, tss,
                                   **_parse_kv(args.fields))
                row["nb_points"], row["nb_timestamps"] = len(pts), len(tss)
    except (psycopg.Error, ValueError, KeyError, OSError) as e:
        print(f"Erreur : {e}", file=sys.stderr)
        return 1

    print(json.dumps(row, default=str, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
