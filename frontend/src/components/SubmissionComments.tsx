/**
 * SubmissionComments.tsx — M22 Discussion UI
 *
 * 顯示 submission 的留言串，支援：
 * - 新增留言（一般 / 內部標記）
 * - 回覆（parent_id）
 * - 編輯（5分鐘窗口後仍可編輯，後端決定）
 * - 刪除（作者或特權用戶）
 * - 附件上傳（file picker）
 */

import { useState } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { commentsApi, attachmentsApi, type CommentOut } from '../api/client'

interface Props {
  req_no: string
  currentUserEmail?: string
  isPrivileged?: boolean
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function formatDate(iso: string): string {
  return new Date(iso).toLocaleString('zh-TW', { timeZone: 'Asia/Taipei' })
}

interface CommentFormProps {
  req_no: string
  parentId?: number | null
  onSuccess: () => void
  onCancel?: () => void
  isPrivileged?: boolean
}

function CommentForm({ req_no, parentId, onSuccess, onCancel, isPrivileged }: CommentFormProps) {
  const [body, setBody] = useState('')
  const [isInternal, setIsInternal] = useState(false)
  const [pendingFile, setPendingFile] = useState<File | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!body.trim()) return
    setSubmitting(true)
    setError(null)
    try {
      const comment = await commentsApi.create(req_no, {
        body_markdown: body,
        is_internal: isInternal,
        parent_id: parentId ?? null,
        attachment_ids: [],
      })
      if (pendingFile) {
        await attachmentsApi.upload(req_no, pendingFile, comment.id)
      }
      setBody('')
      setIsInternal(false)
      setPendingFile(null)
      onSuccess()
    } catch (err: unknown) {
      const msg = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail
      setError(msg ?? '送出失敗，請稍後再試')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-2">
      <textarea
        className="w-full border border-gray-300 rounded px-3 py-2 text-sm resize-y min-h-[80px] focus:outline-none focus:ring-2 focus:ring-blue-400"
        placeholder={parentId ? '回覆...' : '新增留言（支援 @mention）'}
        value={body}
        onChange={(e) => setBody(e.target.value)}
        disabled={submitting}
      />
      <div className="flex items-center gap-4 flex-wrap">
        {isPrivileged && (
          <label className="flex items-center gap-1 text-sm text-gray-600 cursor-pointer">
            <input
              type="checkbox"
              checked={isInternal}
              onChange={(e) => setIsInternal(e.target.checked)}
              disabled={submitting}
              className="accent-yellow-500"
            />
            內部備注
          </label>
        )}
        <label className="text-sm text-gray-500 cursor-pointer hover:text-blue-600">
          <input
            type="file"
            className="hidden"
            onChange={(e) => setPendingFile(e.target.files?.[0] ?? null)}
            disabled={submitting}
          />
          {pendingFile ? (
            <span className="text-blue-600">{pendingFile.name} ({formatBytes(pendingFile.size)})</span>
          ) : (
            <span>附加檔案</span>
          )}
        </label>
        <div className="ml-auto flex gap-2">
          {onCancel && (
            <button
              type="button"
              onClick={onCancel}
              disabled={submitting}
              className="px-3 py-1 text-sm text-gray-500 hover:text-gray-700"
            >
              取消
            </button>
          )}
          <button
            type="submit"
            disabled={submitting || !body.trim()}
            className="px-3 py-1 text-sm bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
          >
            {submitting ? '送出中...' : '送出'}
          </button>
        </div>
      </div>
      {error && <p className="text-red-500 text-xs">{error}</p>}
    </form>
  )
}

interface CommentItemProps {
  comment: CommentOut
  req_no: string
  currentUserEmail?: string
  isPrivileged?: boolean
  depth?: number
  onRefresh: () => void
}

function CommentItem({ comment, req_no, currentUserEmail, isPrivileged, depth = 0, onRefresh }: CommentItemProps) {
  const [replying, setReplying] = useState(false)
  const [editing, setEditing] = useState(false)
  const [editBody, setEditBody] = useState(comment.body_markdown)
  const [saving, setSaving] = useState(false)

  const canEdit = currentUserEmail === comment.author_email || isPrivileged
  const isDeleted = !!comment.deleted_at

  const handleEdit = async () => {
    setSaving(true)
    try {
      await commentsApi.edit(comment.id, editBody)
      setEditing(false)
      onRefresh()
    } finally {
      setSaving(false)
    }
  }

  const handleDelete = async () => {
    if (!confirm('確認刪除此留言？')) return
    await commentsApi.delete(comment.id)
    onRefresh()
  }

  const handleDownload = async (attachmentId: number, filename: string) => {
    const blob = await attachmentsApi.download(attachmentId)
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = filename
    a.click()
    URL.revokeObjectURL(url)
  }

  if (isDeleted && depth === 0) {
    return (
      <div className="text-sm text-gray-400 italic pl-4 py-2 border-l-2 border-gray-100">
        [留言已刪除]
      </div>
    )
  }

  return (
    <div className={`${depth > 0 ? 'ml-8 border-l-2 border-gray-100 pl-4' : ''} mb-3`}>
      <div className={`rounded-lg p-3 ${comment.is_internal ? 'bg-yellow-50 border border-yellow-200' : 'bg-gray-50'}`}>
        {/* Header */}
        <div className="flex items-center gap-2 mb-2 flex-wrap">
          <span className="font-medium text-sm text-gray-800">{comment.author_email}</span>
          {comment.is_internal && (
            <span className="text-xs bg-yellow-200 text-yellow-800 px-1.5 py-0.5 rounded">內部</span>
          )}
          <span className="text-xs text-gray-400">{formatDate(comment.created_at)}</span>
          {comment.updated_at && comment.updated_at !== comment.created_at && (
            <span className="text-xs text-gray-400">(已編輯)</span>
          )}
          {canEdit && !isDeleted && (
            <div className="ml-auto flex gap-2">
              <button
                onClick={() => { setEditing(!editing); setEditBody(comment.body_markdown) }}
                className="text-xs text-blue-500 hover:text-blue-700"
              >
                {editing ? '取消編輯' : '編輯'}
              </button>
              <button
                onClick={handleDelete}
                className="text-xs text-red-400 hover:text-red-600"
              >
                刪除
              </button>
            </div>
          )}
        </div>

        {/* Body */}
        {editing ? (
          <div className="space-y-2">
            <textarea
              className="w-full border border-gray-300 rounded px-2 py-1 text-sm resize-y min-h-[60px]"
              value={editBody}
              onChange={(e) => setEditBody(e.target.value)}
              disabled={saving}
            />
            <div className="flex gap-2">
              <button
                onClick={handleEdit}
                disabled={saving || !editBody.trim()}
                className="px-2 py-1 text-xs bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
              >
                {saving ? '儲存中...' : '儲存'}
              </button>
              <button
                onClick={() => setEditing(false)}
                className="px-2 py-1 text-xs text-gray-500 hover:text-gray-700"
              >
                取消
              </button>
            </div>
          </div>
        ) : (
          <p className="text-sm text-gray-700 whitespace-pre-wrap">{comment.body_markdown}</p>
        )}

        {/* Attachments */}
        {comment.attachments.length > 0 && (
          <div className="mt-2 space-y-1">
            {comment.attachments.map((att) => (
              <button
                key={att.id}
                onClick={() => handleDownload(att.id, att.filename)}
                className="flex items-center gap-2 text-xs text-blue-600 hover:text-blue-800 hover:underline"
              >
                <span>📎</span>
                <span>{att.filename}</span>
                <span className="text-gray-400">({formatBytes(att.size_bytes)})</span>
              </button>
            ))}
          </div>
        )}

        {/* Reply button */}
        {!isDeleted && depth < 2 && (
          <button
            onClick={() => setReplying(!replying)}
            className="mt-2 text-xs text-gray-400 hover:text-blue-600"
          >
            {replying ? '取消回覆' : '回覆'}
          </button>
        )}

        {/* Reply form */}
        {replying && (
          <div className="mt-2">
            <CommentForm
              req_no={req_no}
              parentId={comment.id}
              onSuccess={() => { setReplying(false); onRefresh() }}
              onCancel={() => setReplying(false)}
              isPrivileged={isPrivileged}
            />
          </div>
        )}
      </div>

      {/* Replies */}
      {comment.replies.length > 0 && (
        <div className="mt-2">
          {comment.replies.map((reply) => (
            <CommentItem
              key={reply.id}
              comment={reply}
              req_no={req_no}
              currentUserEmail={currentUserEmail}
              isPrivileged={isPrivileged}
              depth={depth + 1}
              onRefresh={onRefresh}
            />
          ))}
        </div>
      )}
    </div>
  )
}

export default function SubmissionComments({ req_no, currentUserEmail, isPrivileged }: Props) {
  const queryClient = useQueryClient()

  const { data: comments, isLoading, isError } = useQuery({
    queryKey: ['comments', req_no],
    queryFn: () => commentsApi.list(req_no),
    staleTime: 30_000,
  })

  const refresh = () => queryClient.invalidateQueries({ queryKey: ['comments', req_no] })

  if (isLoading) {
    return <div className="text-sm text-gray-400 py-4">載入留言中...</div>
  }

  if (isError) {
    return <div className="text-sm text-red-500 py-4">無法載入留言</div>
  }

  const topLevel = (comments ?? []).filter((c) => !c.parent_id)

  return (
    <div className="space-y-4">
      <h3 className="font-semibold text-gray-700">
        討論 {comments && comments.length > 0 ? `(${comments.length})` : ''}
      </h3>

      {topLevel.length === 0 ? (
        <p className="text-sm text-gray-400">尚無留言</p>
      ) : (
        <div>
          {topLevel.map((comment) => (
            <CommentItem
              key={comment.id}
              comment={comment}
              req_no={req_no}
              currentUserEmail={currentUserEmail}
              isPrivileged={isPrivileged}
              depth={0}
              onRefresh={refresh}
            />
          ))}
        </div>
      )}

      <div className="pt-2 border-t border-gray-100">
        <CommentForm
          req_no={req_no}
          onSuccess={refresh}
          isPrivileged={isPrivileged}
        />
      </div>
    </div>
  )
}
