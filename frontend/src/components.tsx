import type { ReactNode } from "react";

export function PageHeading({
  eyebrow,
  title,
  description,
  action,
}: {
  eyebrow: string;
  title: string;
  description: string;
  action?: ReactNode;
}) {
  return (
    <div className="page-heading">
      <div>
        <div className="eyebrow">{eyebrow}</div>
        <h1>{title}</h1>
        <p>{description}</p>
      </div>
      {action}
    </div>
  );
}

export function Panel({
  title,
  subtitle,
  children,
  className = "",
  action,
}: {
  title: string;
  subtitle?: string;
  children: ReactNode;
  className?: string;
  action?: ReactNode;
}) {
  return (
    <section className={`panel ${className}`}>
      <div className="panel-heading">
        <div>
          <h2>{title}</h2>
          {subtitle && <p>{subtitle}</p>}
        </div>
        {action}
      </div>
      {children}
    </section>
  );
}

export function MetricCard({
  label,
  value,
  note,
  accent,
  icon,
}: {
  label: string;
  value: string;
  note: string;
  accent: string;
  icon: string;
}) {
  return (
    <article className="metric-card">
      <span className="metric-accent" style={{ backgroundColor: accent }} />
      <span className="metric-card-icon" style={{ color: accent }}>{icon}</span>
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>
      <div className="metric-note">{note}</div>
    </article>
  );
}

export function EmptyState({
  message,
  icon = "⌁",
}: {
  message: string;
  icon?: string;
}) {
  return (
    <div className="empty-chart">
      <span className="empty-chart-icon">{icon}</span>
      <p>{message}</p>
    </div>
  );
}

export function SelectField({
  label,
  value,
  options,
  onChange,
  disabled = false,
}: {
  label: string;
  value: string;
  options: { value: string; label: string }[];
  onChange: (value: string) => void;
  disabled?: boolean;
}) {
  return (
    <label className="filter-control">
      <span>{label}</span>
      <select
        disabled={disabled || options.length === 0}
        onChange={(event) => onChange(event.target.value)}
        value={value}
      >
        {options.length
          ? options.map((option) => (
              <option key={option.value} value={option.value}>{option.label}</option>
            ))
          : <option value="">No data available</option>}
      </select>
    </label>
  );
}
