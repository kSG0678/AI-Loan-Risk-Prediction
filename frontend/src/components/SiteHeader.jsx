export default function SiteHeader({ activePage }) {
  return (
    <header className="topbar">
      <a className="brand" href="/" aria-label="Northstar home">
        <span className="brand-mark" aria-hidden="true">
          <svg viewBox="0 0 32 32" fill="none">
            <path d="M16 3.5 18.9 13l9.6 3-9.6 3L16 28.5 13 19l-9.5-3 9.5-3L16 3.5Z" fill="currentColor" />
          </svg>
        </span>
        <span className="brand-name">northstar<span>.</span></span>
      </a>
      <nav className="site-nav" aria-label="Main navigation">
        <a
          href="/"
          aria-current={activePage === "assessment" ? "page" : undefined}
        >
          New assessment
        </a>
        <a
          href="/history"
          aria-current={activePage === "history" ? "page" : undefined}
        >
          Prediction history
        </a>
      </nav>
      <div className="topbar-note">
        <span className="live-dot" />
        <span>AI-powered loan assessment</span>
      </div>
    </header>
  );
}
