import { NextResponse } from "next/server";
import path from "node:path";
import fs from "node:fs/promises";
import { parse } from "csv-parse";
import type { System, Planet } from "../../data";
export const runtime = "nodejs";

function normKey(s: string) {
  return s.toLowerCase().replace(/[^a-z0-9]+/g, "");
}

function pick<T = any>(row: Record<string, any>, keys: string[]): T | undefined {
  const nmap: Record<string, string> = {};
  for (const k of Object.keys(row)) nmap[normKey(k)] = k;
  for (const k of keys) {
    const nk = normKey(k);
    if (nmap[nk] !== undefined) return row[nmap[nk]] as T;
  }
  return undefined;
}

function toNum(v: any): number | undefined {
  if (v === null || v === undefined) return undefined;
  if (typeof v === "number") return isFinite(v) ? v : undefined;
  const s = String(v).trim();
  if (!s || s.toLowerCase() === "na" || s.toLowerCase() === "nan") return undefined;
  const x = Number(s);
  return isFinite(x) ? x : undefined;
}

export async function GET() {
  try {

    const dataDir = path.resolve(process.cwd(), "../../data/clean");
    let files: string[] = [];
    try {
      const entries = await fs.readdir(dataDir, { withFileTypes: true });
      files = entries
        .filter((e) => e.isFile() && e.name.toLowerCase().endsWith(".csv"))
        .map((e) => path.join(dataDir, e.name));
    } catch (e) {

      return NextResponse.json([] as System[]);
    }

    if (!files.length) return NextResponse.json([] as System[]);


    const prefer = files.filter((f) => /(filtered)/i.test(path.basename(f)));
    const targets = prefer.length ? prefer : files;

    const systemsMap = new Map<string, System>();

    for (const file of targets) {
      const raw = await fs.readFile(file, "utf8");

      const text = raw.replace(/^\uFEFF/, "");
      let records: Record<string, any>[] = [];
      try {
        records = await new Promise((resolve, reject) =>
          parse(
            text,
            {
              columns: true,
              skip_empty_lines: true,
              relax_column_count: true,
              trim: true,
            },
            (err, output) => (err ? reject(err) : resolve(output as any))
          )
        );
      } catch (e) {
        continue;
      }

      for (const row of records) {

        let host =
          (pick<string>(row, [
            "hostname",
            "host_name",
            "system",
            "systemname",
            "star_name",
            "stellarname",
            "name",
          ]) || "").toString().trim();

        const fullPlName = (pick<string>(row, ["pl_name", "planetname"]) || "").toString().trim();
        if (!host && fullPlName) {

          host = fullPlName.replace(/\s+[bcdefghijklmnopqrsuvwxyz].*$/i, "").trim();
        }
        if (!host) continue;


        const teff = toNum(pick(row, ["st_teff", "teff", "stteff", "stellar_teff"])) ?? 5800;
        const stRad = toNum(pick(row, ["st_rad", "st_radius", "starradius", "st_rad_rs"])) ?? undefined;
        const stMass = toNum(pick(row, ["st_mass", "st_mass_ms", "starmass"])) ?? undefined;
        

        const ra = toNum(pick(row, ["ra", "ra_deg", "st_ra"])) ?? undefined;
        const dec = toNum(pick(row, ["dec", "dec_deg", "st_dec"])) ?? undefined;
        const distance = toNum(pick(row, ["sy_dist", "st_dist", "distance", "distance_pc"])) ?? undefined;


        let letter = (pick<string>(row, ["pl_letter", "planetletter"]) || "").toString().trim();
        if (!letter && fullPlName) {
          const m = fullPlName.match(/\s([a-z])(?!.*[a-z]).*$/i);
          if (m) letter = m[1].toLowerCase();
        }
        const a_au = toNum(pick(row, ["pl_orbsmax", "a_au", "a", "sma", "sma_au"])) ?? undefined;
        const period_days = toNum(pick(row, ["pl_orbper", "period", "period_days", "p", "orbitalperiod"])) ?? undefined;
        const e = toNum(pick(row, ["pl_orbeccen", "e", "ecc", "eccentricity"])) ?? undefined;
        const radius_re = toNum(pick(row, ["pl_rade", "radius_re", "radius_earth"])) ?? undefined;
        const radius_rj = toNum(pick(row, ["pl_radj", "radius_rj", "radius_jupiter"])) ?? undefined;
        const incl = toNum(pick(row, ["pl_orbincl", "incl", "inclination"])) ?? undefined;
        const pl_eqt = toNum(pick(row, ["pl_eqt", "eqt", "equilibrium_temp"])) ?? undefined;


        if (a_au === undefined || period_days === undefined) continue;

        const sys = systemsMap.get(host) ?? {
          name: host,
          star: { teff, radius_rs: stRad, mass_ms: stMass },
          planets: [],
          ra,
          dec,
          distance_pc: distance,
        };

        if (!systemsMap.has(host)) {
          systemsMap.set(host, sys);
        } else {
          sys.star.teff = sys.star.teff || teff;
          if (sys.star.radius_rs === undefined) sys.star.radius_rs = stRad;
          if (sys.star.mass_ms === undefined) sys.star.mass_ms = stMass;
          if (sys.ra === undefined) sys.ra = ra;
          if (sys.dec === undefined) sys.dec = dec;
          if (sys.distance_pc === undefined) sys.distance_pc = distance;
        }

        const planet: Planet = {
          name: letter || (fullPlName ? fullPlName.replace(/^.*\s/, "") : ""),
          a_au,
          period_days,
          e,
          radius_re,
          radius_rj,
          incl,
          pl_eqt,
        };
        sys.planets.push(planet);
      }
    }

    const systems: System[] = Array.from(systemsMap.values())
      .map((s) => ({
        ...s,
        planets: s.planets.sort((a, b) => (a.a_au ?? 0) - (b.a_au ?? 0)),
      }))
      .filter((s) => s.planets.length > 0)
      .sort((a, b) => a.name.localeCompare(b.name));

    return NextResponse.json(systems);
  } catch (err) {
    console.error("/api/systems error", err);
    return NextResponse.json({ error: "Failed to load systems" }, { status: 500 });
  }
}
