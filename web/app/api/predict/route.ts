import { NextResponse } from 'next/server'
import { DEFAULT_VITALS, predictRisk } from '@/lib/inference'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'

const CORS_HEADERS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type',
}

/**
 * POST /api/predict
 * Body: patient vitals JSON. Returns risk class, alert priority and class
 * probabilities from the trained Logistic Regression model.
 */
export async function POST(request: Request) {
  let payload: unknown

  try {
    payload = await request.json()
  } catch {
    return NextResponse.json(
      { error: 'Invalid JSON body' },
      { status: 400, headers: CORS_HEADERS },
    )
  }

  if (typeof payload !== 'object' || payload === null || Array.isArray(payload)) {
    return NextResponse.json(
      { error: 'Request body must be a JSON object of vitals' },
      { status: 400, headers: CORS_HEADERS },
    )
  }

  try {
    const result = predictRisk(payload as Record<string, unknown>)
    return NextResponse.json(result, { headers: CORS_HEADERS })
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Prediction failed'
    return NextResponse.json({ error: message }, { status: 500, headers: CORS_HEADERS })
  }
}

/**
 * GET /api/predict — health check that also exercises the model.
 */
export async function GET() {
  return NextResponse.json(
    {
      status: 'ok',
      model: 'LogisticRegression (multinomial, lbfgs)',
      test_accuracy: 0.9703,
      sample_prediction: predictRisk(DEFAULT_VITALS),
    },
    { headers: CORS_HEADERS },
  )
}

export async function OPTIONS() {
  return new NextResponse(null, { status: 204, headers: CORS_HEADERS })
}
