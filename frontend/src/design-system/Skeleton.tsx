interface SkeletonRectProps {
  className?: string
  height?: string | number
  width?: string | number
}

export function SkeletonRect({ className = '', height = '1.25rem', width = '100%' }: SkeletonRectProps) {
  return (
    <div
      className={`animate-pulse rounded bg-gray-200 ${className}`}
      style={{
        height: typeof height === 'number' ? `${height}px` : height,
        width: typeof width === 'number' ? `${width}px` : width,
      }}
    />
  )
}

export function SkeletonKpiCard() {
  return (
    <div className="bg-white rounded-xl shadow-sm p-4 border border-gray-100">
      <SkeletonRect height={12} width="60%" className="mb-3" />
      <SkeletonRect height={36} width="50%" className="mb-2" />
      <SkeletonRect height={10} width="80%" />
    </div>
  )
}
