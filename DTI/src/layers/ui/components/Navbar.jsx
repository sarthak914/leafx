import { NavLink } from "react-router-dom";
import { useTranslation } from "react-i18next";

function Navbar() {
  const { t } = useTranslation();

  const links = [
    { to: "/", label: t("nav_home") },
    { to: "/scan", label: t("nav_scan") },
    { to: "/results", label: t("nav_results") },
    { to: "/history", label: t("nav_history") },
  ];

  return (
    <header className="sticky top-0 z-40 border-b border-white/10 bg-brand/80 backdrop-blur-xl">
      <nav className="page-shell flex items-center justify-between py-4">
        <p className="text-sm font-bold tracking-[0.12em] sm:text-base">{t("brand_name")}</p>
        <div className="hidden items-center gap-2 md:flex">
          {links.map((link) => (
            <NavLink
              key={link.to}
              to={link.to}
              end={link.to === "/"}
              className={({ isActive }) =>
                `rounded-full px-4 py-2 text-xs tracking-[0.12em] transition ${
                  isActive
                    ? "border border-leafPrimary/50 bg-leafPrimary/10 text-leafPrimary shadow-glowPrimary"
                    : "border border-transparent text-slate-300 hover:border-white/10 hover:bg-white/5"
                }`
              }
            >
              {link.label}
            </NavLink>
          ))}
        </div>
      </nav>
    </header>
  );
}

export default Navbar;
