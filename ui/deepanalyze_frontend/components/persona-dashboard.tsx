"use client";

import React, { useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
} from "recharts";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Users, AlertTriangle, Coins } from "lucide-react";

export interface ProfileAttributes {
  high_spender_pct?: number;
  avg_fee?: number;
  tier_upgrade_rate?: number;
  tier_downgrade_rate?: number;
  usage_decline_strong_pct?: number;
  usage_decline_mild_pct?: number;
  usage_unstable_pct?: number;
  status_worsening_pct?: number;
  loyalty_rank_avg?: number;
  csat_avg?: number;
  ces_avg?: number;
  package_composition?: Record<string, number>;
  service_composition?: Record<string, number>;
}

export interface Persona {
  cluster_id: number;
  persona_name: string;
  support: number;
  support_pct: number;
  arpu: number;
  churn_rate: number;
  confidence: string;
  sample_persona_text: string;
  severity?: string;
  risk?: string;
  risk_tier?: string;
  persona_type?: string;
  feature_means?: Record<string, number>;
  evidence?: Record<string, number>;
  profile_attributes?: ProfileAttributes;
  recommended_actions?: string[];
  domain_signature?: Record<string, { stars: number; top_features: [string, number, number][] }>;
  churn_driver?: string;
  /** Measured share of each outcome inside this persona's cluster; {} when the dataset
   *  carries no status column. Never inferred — see pipeline.cohort_mix. */
  cohort_mix?: Record<string, number>;
  /** Measured share still a customer, or null/undefined when nobody named which status
   *  values mean "active". Absent is NOT the same as zero. */
  active_pct?: number | null;
  /** How this group is distributed across each categorical column, ALWAYS carrying the
   *  share the same value holds across the whole dataset. A share on its own reads as a
   *  finding: 18.6% of one group in Ha Noi looks like concentration until you see Ha Noi
   *  is 16.5% of the file. `value: null` is the remainder of cells too small to name.
   *  See pipeline.category_mix. */
  category_mix?: Record<string, CategoryShare[]>;
  /** Business label per categorical column, so the UI shows "Khu vực" not "LOCATIONNAME". */
  category_labels?: Record<string, string>;
}

export interface CategoryShare {
  /** null means "everything below the naming floor", pooled — see pipeline._MIN_CATEGORY_CELL. */
  value: string | null;
  rows: number;
  share: number;
  dataset_share: number | null;
  /** share / dataset_share, or null when the value is absent from the dataset. */
  lift: number | null;
}

interface PersonaDashboardProps {
  data: Persona[];
}

const COLORS = ["#0088FE", "#00C49F", "#FFBB28", "#FF8042", "#8884d8", "#82ca9d"];

export function PersonaDashboard({ data }: PersonaDashboardProps) {
  const [activeTab, setActiveTab] = useState<"overview" | "churn" | "revenue">("overview");

  let actualData = data;
  if (data && !Array.isArray(data) && Array.isArray((data as any).personas)) {
    actualData = (data as any).personas;
  }

  if (!actualData || !Array.isArray(actualData)) {
    return (
      <div className="p-4 border border-yellow-500 rounded text-yellow-700 bg-yellow-50 text-sm">
        Waiting for valid Persona JSON data format...
      </div>
    );
  }

  // Datasets without real ARPU/churn ground truth (e.g. no `rmdt` or `arpu` column) never get
  // these fields populated by the pipeline — treat missing/non-finite values as 0 instead of
  // letting `undefined` arithmetic poison the whole weighted average into NaN.
  const safeNum = (v: unknown): number => (typeof v === "number" && isFinite(v) ? v : 0);
  const hasChurnData = actualData.some((item) => typeof item.churn_rate === "number" && isFinite(item.churn_rate));
  const hasRevenueData = actualData.some((item) => typeof item.arpu === "number" && isFinite(item.arpu) && item.arpu > 0);

  // The tile printed "N/A — Không có dữ liệu ARPU" while every persona card below it showed
  // "Cước phí trung bình: 198.932". Both were right about their own field: revenue-at-risk
  // needs `arpu`, which the POST_CHURN path never sets, while the fee lives in
  // profile_attributes.avg_fee. Rather than claim the fee is missing, report the fee — and
  // do not call it revenue at risk, because a cohort that has already left puts none at risk.
  const feeWeighted = actualData.reduce(
    (acc, item) => {
      const fee = safeNum((item as any).profile_attributes?.avg_fee);
      const support = safeNum(item.support);
      return fee > 0 ? { total: acc.total + fee * support, weight: acc.weight + support } : acc;
    },
    { total: 0, weight: 0 },
  );
  const avgFee = feeWeighted.weight > 0 ? feeWeighted.total / feeWeighted.weight : 0;
  // POST_CHURN datasets: every persona is already-churned by definition, so churn_rate=1.0 (100%)
  // for all of them is CORRECT data, not a bug — but showing a bare "100%" in a "Weighted average"
  // KPI tile reads as an alarming, out-of-context metric. Detect via churn_driver (only ever set
  // for POST_CHURN mode) and relabel the tile instead of treating it as future-risk framing.
  const isPostChurnDataset = actualData.some((item) => Boolean((item as any).churn_driver));

  // The tile used to print a hardcoded "100% — Toàn bộ mẫu đã rời mạng" for every
  // POST_CHURN dataset. Nothing counted that 100%: it followed from one persona having a
  // churn_driver. On the 62,467-row Churn_VT export it was wrong by 4,533 subscribers who
  // had restored service — 92.7%, not 100%. When the pipeline was given a status column it
  // now reports what it MEASURED, weighted by cluster size; with no status column we say
  // the proportion is unknown rather than invent one.
  const cohortTotals = actualData.reduce<Record<string, number>>((acc, item) => {
    const mix = item.cohort_mix;
    if (!mix) return acc;
    const support = safeNum(item.support);
    for (const [status, share] of Object.entries(mix)) {
      acc[status] = (acc[status] ?? 0) + safeNum(share) * support;
    }
    return acc;
  }, {});
  const cohortMeasured = Object.values(cohortTotals).reduce((a, b) => a + b, 0);
  const cohortShares = Object.entries(cohortTotals)
    .map(([status, weight]) => [status, weight / cohortMeasured] as const)
    .sort((a, b) => b[1] - a[1]);

  // Calculate Revenue at Risk
  const chartData = actualData.map((item) => {
    const arpu = safeNum(item.arpu);
    const churn_rate = safeNum(item.churn_rate);
    return {
      ...item,
      arpu,
      churn_rate,
      total_revenue: item.support * arpu,
      revenue_at_risk: item.support * arpu * churn_rate,
      // Add C{id} prefix to guarantee unique keys for Recharts, preventing duplicate X-Axis labels from overwriting each other
      short_name: `C${item.cluster_id}: ${item.persona_name.length > 12 ? item.persona_name.substring(0, 12) + "..." : item.persona_name}`,
    };
  });

  const totalSupport = actualData.reduce((acc, curr) => acc + curr.support, 0);
  const totalRevenueAtRisk = chartData.reduce((acc, curr) => acc + curr.revenue_at_risk, 0);
  const avgChurn = actualData.reduce((acc, curr) => acc + safeNum(curr.churn_rate) * curr.support, 0) / (totalSupport || 1);

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat("vi-VN", { style: "currency", currency: "VND" }).format(value);
  };

  const formatPercent = (value: number) => {
    return (value * 100).toFixed(1) + "%";
  };

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white dark:bg-gray-800 p-3 border border-gray-200 dark:border-gray-700 rounded-md shadow-md text-xs z-50">
          <p className="font-bold mb-1 text-gray-900 dark:text-gray-100">{label}</p>
          {payload.map((entry: any, index: number) => (
            <p key={index} style={{ color: entry.color }} className="my-1">
              {entry.name}: {entry.name.includes("Revenue") || entry.name.includes("ARPU")
                ? formatCurrency(entry.value)
                : entry.name.includes("Churn") || entry.name.includes("Rate")
                  ? formatPercent(entry.value)
                  : entry.value}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="w-full my-6 space-y-4 font-sans">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Customers</CardTitle>
            <Users className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{totalSupport.toLocaleString()}</div>
            <p className="text-xs text-muted-foreground">Clustered in {data.length} personas</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">{isPostChurnDataset ? "Churn Status" : "Avg Churn Rate"}</CardTitle>
            <AlertTriangle className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            {cohortMeasured > 0 ? (
              <>
                <div className="text-2xl font-bold">{formatPercent(cohortShares[0][1])}</div>
                <p className="text-xs text-muted-foreground">
                  {cohortShares.map(([status, share]) => `${status} ${formatPercent(share)}`).join(" · ")}
                </p>
              </>
            ) : isPostChurnDataset ? (
              <>
                <div className="text-2xl font-bold">—</div>
                <p className="text-xs text-muted-foreground">
                  Dữ liệu mang dấu hiệu sau rời mạng; tỉ lệ chưa đo được vì không có cột trạng thái
                </p>
              </>
            ) : (
              <>
                <div className="text-2xl font-bold">{hasChurnData ? formatPercent(avgChurn) : "N/A"}</div>
                <p className="text-xs text-muted-foreground">{hasChurnData ? "Weighted average" : "Không có dữ liệu churn"}</p>
              </>
            )}
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">
              {hasRevenueData ? "Total Revenue at Risk" : avgFee > 0 ? "Cước phí trung bình" : "Revenue Data"}
            </CardTitle>
            <Coins className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className={`text-2xl font-bold ${hasRevenueData ? "text-red-500" : ""}`}>
              {hasRevenueData ? formatCurrency(totalRevenueAtRisk) : avgFee > 0 ? formatCurrency(avgFee) : "N/A"}
            </div>
            <p className="text-xs text-muted-foreground">
              {hasRevenueData
                ? "Monthly estimated"
                : avgFee > 0
                  ? "Bình quân theo quy mô nhóm — không phải doanh thu rủi ro"
                  : "Không có dữ liệu cước phí"}
            </p>
          </CardContent>
        </Card>
      </div>

      <Card className="w-full">
        <CardHeader>
          <CardTitle>Dynamic Persona Analysis</CardTitle>
          <CardDescription>Interactive visualizations of the generated personas</CardDescription>
          <div className="flex space-x-2 mt-4">
            <Badge
              variant={activeTab === "overview" ? "default" : "outline"}
              className="cursor-pointer"
              onClick={() => setActiveTab("overview")}
            >
              Population Overview
            </Badge>
            {hasChurnData && (
              <Badge
                variant={activeTab === "churn" ? "default" : "outline"}
                className="cursor-pointer"
                onClick={() => setActiveTab("churn")}
              >
                Churn Risk (Radar)
              </Badge>
            )}
            {hasRevenueData && (
              <Badge
                variant={activeTab === "revenue" ? "default" : "outline"}
                className="cursor-pointer"
                onClick={() => setActiveTab("revenue")}
              >
                Revenue at Risk
              </Badge>
            )}
          </div>
        </CardHeader>
        <CardContent>
          <div className="h-80 w-full mt-4">
            {activeTab === "overview" && (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                  <XAxis
                    dataKey="short_name"
                    angle={-45}
                    textAnchor="end"
                    height={70}
                    tick={{ fontSize: 11 }}
                  />
                  <YAxis />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend verticalAlign="top" />
                  <Bar dataKey="support" name="Customer Count" fill="#8884d8" radius={[4, 4, 0, 0]}>
                    {chartData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            )}

            {activeTab === "churn" && hasChurnData && (
              <ResponsiveContainer width="100%" height="100%">
                <RadarChart cx="50%" cy="50%" outerRadius="80%" data={chartData}>
                  <PolarGrid opacity={0.3} />
                  <PolarAngleAxis dataKey="short_name" tick={{ fontSize: 11 }} />
                  <PolarRadiusAxis angle={30} domain={[0, 1]} tickFormatter={formatPercent} />
                  <Radar name="Churn Rate" dataKey="churn_rate" stroke="#ff4d4f" fill="#ff4d4f" fillOpacity={0.6} />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend />
                </RadarChart>
              </ResponsiveContainer>
            )}

            {activeTab === "revenue" && hasRevenueData && (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                  <XAxis
                    dataKey="short_name"
                    angle={-45}
                    textAnchor="end"
                    height={70}
                    tick={{ fontSize: 11 }}
                  />
                  <YAxis tickFormatter={(val) => `${(val / 1000000).toFixed(0)}M`} />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend verticalAlign="top" />
                  <Bar dataKey="total_revenue" name="Safe Revenue" stackId="a" fill="#10b981" />
                  <Bar dataKey="revenue_at_risk" name="Revenue at Risk" stackId="a" fill="#ef4444" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
